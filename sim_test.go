package main

// A simulated R730xd for the real controller: CPU temperatures with two time constants (die ~20 s,
// heatsinks/air ~150 s) towards the fitted model's steady state, random CPU power bursts clipped by
// the power cap, fans that take a few seconds to change speed, the iDRAC's lagging wall power, and
// whole-degree coretemp readings with a little noise. controller.cycle() runs every 5 simulated
// seconds against it. Run: FANCTL_SIM=1 go test -run TestSim -v  (FANCTL_SIM_CSV=dir writes traces).

import (
	"fmt"
	"io"
	"log"
	"math"
	"math/rand"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"testing"
	"time"
)

// the server's live settings (02-cluster/host/fanctl/fanctl.toml, 2026-11-11)
const simConfig = `
mode = "model"
[shutdown]
cpu = 87
hdd = 57
exhaust = 65
[safe]
duty = 20
cap = 230
clamp = true
check_duty = 15
[model]
min_duty = 10
fan_max = 20
cpu_target = 80
margin = 3
cpu_limit = 83
emergency_duty = 25
warm_curve = [[77, 10], [80, 15], [83, 20]]
hdd_curve = [[50, 10], [53, 25], [55, 45]]
exhaust_model = [0.3716, 0.1911]
exhaust_excess_curve = [[5, 10], [7, 20], [9, 40]]
cpu_curve = [[55, 10], [62, 15], [68, 25], [74, 40], [80, 60]]
exhaust_curve = [[45, 10], [50, 20], [54, 40]]
cpu_models = [[7.0, 0.0902, 0.7879, 1.4572, 1.0], [74.7, 0.1608, 1.5213, 1.8038, 1.0]]
fan_rpm_table = [[10, 3240], [15, 4080], [20, 4800], [25, 5560], [30, 6420]]
[power]
cap_min = 230
[inputs]
required_temps = ["CPU1", "CPU2", "Inlet", "Exhaust"]
hdd_min = 0
fan_min_rpm = 1000
interval = "5s"
hdd_interval = "300s"
unknown_after = "15s"
`

type simScenario struct {
	name       string
	offset     float64 // the plant runs this much hotter than the fitted model (model error)
	base       float64 // CPU package power without bursts, W (both sockets)
	stepTo     float64 // after the warm-up the base load steps to this (0 = stays)
	jitter     float64 // second-to-second CPU power noise, W (AR(1), ~3 s correlation)
	dieShare   float64 // share of the rise that follows power within dieTau (the package sensor sits on the die)
	dieTau     float64
	capTau     float64    // how fast the iDRAC enforces its cap, s
	margin     float64    // model.margin for this run (3 = the server's: aims 77, corrects toward 80)
	randomBase [2]float64 // when set: the base load jumps to a random level in this range every ~2 min
	inlet      float64    // inlet air, C (0 = 28)
	// adversarial: where the real machine differs from what the controller believes
	fanEff     float64 // fans this effective compared with the model (0 = 1)
	inletGain  float64 // the CPU follows inlet changes this strongly (0 = 1, the model's assumption)
	raplScale  float64 // RAPL reads this fraction of the real CPU power (0 = 1)
	wallLag    float64 // the iDRAC's wall reading time constant, s (0 = 15)
	bias       float64 // coretemp reads this much off (negative = reads low)
	fanTau     float64 // fans take this long to change speed, s (0 = 3)
	square     float64 // > 0: the base load hops between base and stepTo every this many seconds
	evil       bool    // a burst lands right after every fan decrease
	burstEvery float64 // mean seconds between bursts
	burstW     [2]float64
	burstS     [2]float64
}

type plant struct {
	sc        simScenario
	rng       *rand.Rand
	t         float64
	duty      string // what the controller last commanded ("auto" or %)
	cap       int
	rpm       float64
	cpuW      float64 // actual CPU package power
	wallAvg   float64 // the iDRAC's lagging wall reading
	fast      [2]float64
	slow      [2]float64
	burstEnd  float64
	burstAmp  float64
	nextBurst float64
	bursts    int
	die       [2]float64
	noise     float64
	level     float64
	nextLevel float64
}

const simNonCPU = 139.0 // idle: 224 W wall with ~85 W CPU
const simWarmup = 900.0 // seconds simulated before anything is counted

func (p *plant) wall() float64 { return simNonCPU + p.cpuW }

func (p *plant) in() float64 {
	if p.sc.inlet > 0 {
		return p.sc.inlet
	}
	return 28
}

func (p *plant) tss(cpu int) float64 {
	m := cpuModels[cpu]
	fe, g := p.sc.fanEff, p.sc.inletGain
	if fe == 0 {
		fe = 1
	}
	if g == 0 {
		g = 1
	}
	in := 28 + g*(p.in()-28)
	// weaker fans: the fan-dependent part of the cooling only fe as effective, the rest as at 10 %
	return in + (p.wall()-m.pb)*(m.r0+fe*m.k*math.Pow(p.rpm/1000, -m.n)+(1-fe)*m.k*math.Pow(3.24, -m.n))*m.scale + p.sc.offset
}

func (p *plant) temp(cpu int) float64 { return p.in() + p.die[cpu] + p.fast[cpu] + p.slow[cpu] }

func (p *plant) step(dt float64) {
	p.t += dt
	if p.t >= p.nextBurst { // a burst: more CPU power for a few seconds
		p.burstAmp = p.sc.burstW[0] + p.rng.Float64()*(p.sc.burstW[1]-p.sc.burstW[0])
		p.burstEnd = p.t + p.sc.burstS[0] + p.rng.Float64()*(p.sc.burstS[1]-p.sc.burstS[0])
		p.nextBurst = p.t + p.rng.ExpFloat64()*p.sc.burstEvery
		p.bursts++
	}
	base := p.sc.base
	if p.sc.stepTo > 0 && p.t >= simWarmup {
		base = p.sc.stepTo
		if p.sc.square > 0 && int(p.t/p.sc.square)%2 == 1 {
			base = p.sc.base
		}
	}
	if r := p.sc.randomBase; r[1] > 0 {
		if p.t >= p.nextLevel {
			p.level = r[0] + p.rng.Float64()*(r[1]-r[0])
			p.nextLevel = p.t + 30 + p.rng.ExpFloat64()*90
		}
		base = p.level
	}
	p.noise = p.noise*math.Exp(-dt/3) + p.sc.jitter*math.Sqrt(1-math.Exp(-2*dt/3))*p.rng.NormFloat64()
	demand := base + 8*math.Sin(p.t/400) + p.noise
	if p.t < p.burstEnd {
		demand += p.burstAmp
	}
	allowed := float64(p.cap) - simNonCPU
	if p.cap == 0 {
		allowed = 1e9
	}
	target := math.Min(demand, allowed)
	if demand < target { // the load itself changes at once; only the cap's clipping takes time
		target = demand
	}
	capTau := p.sc.capTau
	if capTau == 0 {
		capTau = 1.5
	}
	if target < p.cpuW && demand > target { // being clipped by the cap: the iDRAC is slow
		p.cpuW += (target - p.cpuW) * math.Min(1, dt/capTau)
	} else {
		p.cpuW = target
	}
	wl := p.sc.wallLag
	if wl == 0 {
		wl = 15
	}
	p.wallAvg += (p.wall() - p.wallAvg) * dt / wl // the iDRAC's reading lags
	want := 5000.0
	if pct, err := strconv.Atoi(p.duty); err == nil {
		want = pctToRPM(float64(pct))
	}
	ft := p.sc.fanTau
	if ft == 0 {
		ft = 3
	}
	p.rpm += (want - p.rpm) * math.Min(1, dt/ft) // fans spin up/down over a few seconds
	for c := 0; c < 2; c++ {
		rise := p.tss(c) - p.in()
		ds := p.sc.dieShare
		p.die[c] += (ds*rise - p.die[c]) * math.Min(1, dt/math.Max(p.sc.dieTau, dt))
		p.fast[c] += ((0.4-ds/2)*rise - p.fast[c]) * dt / 20
		p.slow[c] += ((0.6-ds/2)*rise - p.slow[c]) * dt / 150
	}
}

type plantBMC struct {
	p       *plant
	sensors []Sensor
}

func (b *plantBMC) Sensors() ([]Sensor, error) { return b.sensors, nil }
func (b *plantBMC) Reading(s Sensor) (float64, bool, string, error) {
	p := b.p
	switch k := sensorKey(s); {
	case s.Type == sensorFan:
		off := map[string]float64{"Fan1": -60, "Fan2": 60, "Fan3": 60, "Fan4": -60, "Fan5": -60, "Fan6": 60}[k]
		return math.Round((p.rpm+off)/120) * 120, true, "", nil
	case k == "power":
		return math.Round(p.wallAvg/14) * 14, true, "", nil // the iDRAC reports in 14 W steps
	case k == "Inlet":
		return p.in(), true, "", nil
	case k == "Exhaust":
		hot := math.Max(p.temp(0), p.temp(1))
		return math.Round(p.in() + 0.3716*(hot-p.in()) + 0.1911*(44-p.in()) + 2), true, "", nil
	}
	return 0, false, "unavailable", nil
}
func (b *plantBMC) FanDuty(pct int) error { b.p.duty = strconv.Itoa(pct); return nil }
func (b *plantBMC) FanAuto() error        { b.p.duty = "auto"; return nil }
func (b *plantBMC) SetPowerCap(w int) error {
	b.p.cap = w
	return nil
}
func (b *plantBMC) GetPowerLimit() (PowerLimit, error) {
	return PowerLimit{Active: b.p.cap > 0, Watts: b.p.cap}, nil
}

type simResult struct {
	sc                     simScenario
	meanDuty, meanWall     float64
	changesPerHour         float64
	dutyHist               map[int]int
	maxT2, over80, over83  float64 // max true CPU2; share of time >= 80, >= 83
	over86                 float64
	emergencies, poweroffs int
	jumps                  map[int]int // CPU2 reading change between cycles
	bursts                 int
	cycles                 int
}

func runSim(t *testing.T, sc simScenario, hours float64, csvDir string) simResult {
	_, _, _ = rig(t) // temp dirs, fake coretemp/intel_pstate, drive state
	loadConfigBytes([]byte(strings.Replace(simConfig, "margin = 3", fmt.Sprintf("margin = %g", sc.margin), 1)))
	st.Lock()
	st.mode = "model"
	st.Unlock()
	d := Drive{ID: 1, Serial: "SIMDRIVE"}
	hdds.Lock()
	hdds.ids, hdds.temps, hdds.at, hdds.scanned = []Drive{d}, map[Drive]float64{d: 44}, t0, true
	hdds.Unlock()

	p := &plant{sc: sc, rng: rand.New(rand.NewSource(1)), duty: "13", cap: 586, rpm: 3800, cpuW: sc.base}
	p.wallAvg = p.wall()
	p.nextBurst = p.rng.ExpFloat64() * sc.burstEvery
	for c := 0; c < 2; c++ { // start at the steady state of the base load
		rise := p.tss(c) - p.in()
		ds := sc.dieShare
		p.die[c], p.fast[c], p.slow[c] = ds*rise, (0.4-ds/2)*rise, (0.6-ds/2)*rise
	}
	b := &plantBMC{p: p, sensors: []Sensor{
		{Type: sensorTemp, Entity: 64, Name: "Inlet Temp"}, {Type: sensorTemp, Entity: 64, Name: "Exhaust Temp"},
		{Type: sensorFan, Name: "Fan1 RPM"}, {Type: sensorFan, Name: "Fan2 RPM"}, {Type: sensorFan, Name: "Fan3 RPM"},
		{Type: sensorFan, Name: "Fan4 RPM"}, {Type: sensorFan, Name: "Fan5 RPM"}, {Type: sensorFan, Name: "Fan6 RPM"},
		{Type: sensorCurrent, Name: "Pwr Consumption"},
	}}
	c := newController(b)
	c.start = t0
	oldR, oldP := raplSource, poweroff
	rs := sc.raplScale
	if rs == 0 {
		rs = 1
	}
	raplSource = func() (float64, bool) { return p.cpuW * rs, true }
	poweroffs := 0
	poweroff = func() error { poweroffs++; return nil }
	log.SetOutput(io.Discard)
	defer func() { raplSource, poweroff = oldR, oldP; log.SetOutput(os.Stderr) }()

	var csv *os.File
	if csvDir != "" {
		csv, _ = os.Create(filepath.Join(csvDir, "sim-"+sc.name+".csv"))
		fmt.Fprintln(csv, "t,cpu1,cpu2,read1,read2,cpu_w,wall,cap,duty,rpm,controller")
		defer csv.Close()
	}
	res := simResult{sc: sc, dutyHist: map[int]int{}, jumps: map[int]int{}}
	prevRead := -1
	prevDuty, sumDuty, sumWall := "", 0.0, 0.0
	steps := int(simWarmup + hours*3600)
	for s := 0; s < steps; s++ {
		p.step(1)
		if s%5 != 0 {
			continue
		}
		now := t0.Add(time.Duration(s) * time.Second)
		if s%300 == 0 { // the drive goroutine's snapshot, every hdd_interval
			hdds.Lock()
			hdds.at = now
			hdds.Unlock()
		}
		for cpu := 0; cpu < 2; cpu++ { // whole-degree coretemp with a little noise
			read := math.Floor(p.temp(cpu) + sc.bias + p.rng.NormFloat64()*0.3)
			if cpu == 1 && float64(s) >= simWarmup {
				if prevRead >= 0 {
					res.jumps[int(read)-prevRead]++
				}
				prevRead = int(read)
			}
			os.WriteFile(filepath.Join(hwmonDir, "hwmon"+strconv.Itoa(cpu+1), "temp1_input"), []byte(strconv.Itoa(int(read*1000))+"\n"), 0o644)
		}
		dutyBefore, _ := strconv.Atoi(p.duty)
		c.cycle(now)
		if dutyAfter, _ := strconv.Atoi(p.duty); sc.evil && dutyAfter < dutyBefore && p.t >= p.burstEnd {
			p.nextBurst = p.t // the load hits right after the fans slowed
		}
		if float64(s) < simWarmup {
			prevDuty = p.duty
			continue
		}
		res.cycles++
		duty, _ := strconv.Atoi(p.duty)
		res.dutyHist[duty]++
		sumDuty += float64(duty)
		sumWall += p.wall()
		if p.duty != prevDuty {
			res.changesPerHour++
			if duty == int(emergencyPct) && c.overLimit >= limitReads {
				res.emergencies++
			}
		}
		prevDuty = p.duty
		t2 := p.temp(1)
		res.maxT2 = math.Max(res.maxT2, t2)
		if t2 >= 80 {
			res.over80++
		}
		if t2 >= 83 {
			res.over83++
		}
		if t2 >= 86 {
			res.over86++
		}
		if csv != nil {
			fmt.Fprintf(csv, "%d,%.2f,%.2f,%d,%d,%.1f,%.1f,%d,%s,%.0f,%s\n", s, p.temp(0), t2,
				int(math.Floor(p.temp(0))), int(math.Floor(t2)), p.cpuW, p.wall(), p.cap, p.duty, p.rpm, c.state)
		}
	}
	n := float64(res.cycles)
	res.meanDuty, res.meanWall = sumDuty/n, sumWall/n
	res.changesPerHour /= hours
	res.over80, res.over83, res.over86 = res.over80/n, res.over83/n, res.over86/n
	res.poweroffs, res.bursts = poweroffs, p.bursts
	return res
}

func TestSim(t *testing.T) {
	if os.Getenv("FANCTL_SIM") != "1" {
		t.Skip("set FANCTL_SIM=1")
	}
	real := func(name string, margin, base, stepTo float64) simScenario { // calibrated to the server's 5 s jumps
		return simScenario{name: name, margin: margin, base: base, stepTo: stepTo, jitter: 25, dieShare: 0.3, dieTau: 2, capTau: 5,
			burstEvery: 90, burstW: [2]float64{30, 80}, burstS: [2]float64{5, 30}}
	}
	random := func(name string, margin float64) simScenario {
		sc := real(name, margin, 100, 0)
		sc.randomBase = [2]float64{60, 230}
		return sc
	}
	extreme := func(name string, margin float64) simScenario {
		sc := real(name, margin, 100, 260) // normal load during the warm-up, then far beyond what the cap allows
		sc.jitter, sc.burstEvery, sc.burstW, sc.burstS = 45, 45, [2]float64{100, 150}, [2]float64{5, 40}
		return sc
	}
	insane := func(name string, margin float64) simScenario { // realistic, just a lot of it
		sc := real(name, margin, 100, 0)
		sc.randomBase, sc.inlet = [2]float64{100, 260}, 30
		return sc
	}
	hot := func(sc simScenario) simScenario { sc.inlet = 30; return sc }
	_ = random // kept for reruns of the earlier scenarios
	_, _, _ = insane, extreme, hot
	adv := func(name string, margin float64, f func(*simScenario)) simScenario {
		sc := real(name, margin, 100, 0)
		sc.randomBase, sc.inlet = [2]float64{100, 260}, 30
		f(&sc)
		return sc
	}
	everything := func(s *simScenario) {
		s.offset, s.fanEff, s.raplScale, s.wallLag, s.bias, s.capTau, s.inlet = 3, 0.7, 0.75, 60, -2, 20, 32
		s.randomBase, s.base, s.stepTo, s.square, s.evil = [2]float64{}, 90, 240, 60, true
		s.burstW = [2]float64{60, 120}
	}
	scenarios := []simScenario{
		adv("A-3C-hotter", 0, func(s *simScenario) { s.offset = 3 }),
		adv("B-5C-hotter", 0, func(s *simScenario) { s.offset = 5 }),
		adv("C-fans-30pct-weaker", 0, func(s *simScenario) { s.fanEff = 0.7 }),
		adv("D-inlet34-gain1.5", 0, func(s *simScenario) { s.inlet, s.inletGain = 34, 1.5 }),
		adv("E-rapl-low-wall-lag", 0, func(s *simScenario) { s.raplScale, s.wallLag = 0.75, 60 }),
		adv("F-coretemp-2C-low", 0, func(s *simScenario) { s.bias = -2 }),
		adv("G-slow-cap-extreme", 0, func(s *simScenario) {
			s.randomBase, s.base, s.stepTo, s.capTau = [2]float64{}, 100, 260, 20
			s.jitter, s.burstEvery, s.burstW, s.burstS = 45, 45, [2]float64{100, 150}, [2]float64{5, 40}
		}),
		adv("H-evil-load", 0, func(s *simScenario) {
			s.randomBase, s.base, s.stepTo, s.square, s.evil = [2]float64{}, 90, 230, 60, true
		}),
		adv("I-everything-aim80", 0, everything),
		adv("I-everything-today", 3, everything),
	}
	if os.Getenv("FANCTL_SIM_SWEEP") == "1" {
		scenarios = nil
		for _, m := range []float64{1, 2} {
			sfx := fmt.Sprintf("-m%g", m)
			scenarios = append(scenarios, extreme("full-load"+sfx, m), insane("insane"+sfx, m),
				adv("F-coretemp-2C-low"+sfx, m, func(s *simScenario) { s.bias = -2 }),
				adv("H-evil-load"+sfx, m, func(s *simScenario) {
					s.randomBase, s.base, s.stepTo, s.square, s.evil = [2]float64{}, 90, 230, 60, true
				}),
				adv("I-everything"+sfx, m, everything))
		}
	}
	for _, sc := range scenarios {
		r := runSim(t, sc, 2, os.Getenv("FANCTL_SIM_CSV"))
		keys := make([]int, 0, len(r.dutyHist))
		for k := range r.dutyHist {
			keys = append(keys, k)
		}
		sort.Ints(keys)
		hist := ""
		for _, k := range keys {
			if share := float64(r.dutyHist[k]) / float64(r.cycles); share >= 0.01 {
				hist += fmt.Sprintf(" %d%%:%.0f%%", k, 100*share)
			}
		}
		jt, big := 0, 0
		for k, v := range r.jumps {
			jt += v
			if k >= 2 || k <= -2 {
				big += v
			}
		}
		_ = big
		t.Logf("%-20s | duty %4.1f%% chg/h %4.0f | emerg %2d poweroff %d | CPU2 true max %5.1f >=83 %4.1f%% >=86 %4.1f%% | wall %3.0f |%s",
			sc.name, r.meanDuty, r.changesPerHour, r.emergencies, r.poweroffs, r.maxT2, 100*r.over83, 100*r.over86, r.meanWall, hist)
	}
}
