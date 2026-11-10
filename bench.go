//go:build linux

package main

// fanctl bench: measures the thermal model. Holds each (fan duty, power cap) point for a while under
// a full CPU load and writes one CSV row every 5 s: per-socket coretemp and RAPL power, wall power,
// inlet, exhaust, fan RPM, hottest drive. Every point change is a step, so the steady temperature and
// the time constants both come out of a curve fit; the points don't have to settle.
//
// Run with fanctl stopped:  sudo systemctl stop fanctl; sudo fanctl bench; sudo systemctl start fanctl
// On every exit (done, abort, Ctrl-C) the load is killed, the fans go to the iDRAC and the cap is lifted.

import (
	"bufio"
	"flag"
	"fmt"
	"log"
	"os"
	"os/exec"
	"os/signal"
	"runtime"
	"slices"
	"sort"
	"strconv"
	"strings"
	"syscall"
	"time"
)

type benchPoint struct{ duty, cap int }

// benchOrder walks the grid in a snake (duties up for one cap, down for the next), so every step is small.
func benchOrder(duties, caps []int) []benchPoint {
	var out []benchPoint
	for i, c := range caps {
		ds := slices.Clone(duties)
		if i%2 == 1 {
			slices.Reverse(ds)
		}
		for _, d := range ds {
			out = append(out, benchPoint{d, c})
		}
	}
	return out
}

func intList(s string) ([]int, error) {
	var out []int
	for _, f := range strings.Split(s, ",") {
		v, err := strconv.Atoi(strings.TrimSpace(f))
		if err != nil {
			return nil, err
		}
		out = append(out, v)
	}
	return out, nil
}

// raplJoules: per-socket package energy counters (µJ) and their wrap ranges.
func raplJoules() (e, rng []int64) {
	for _, dom := range []string{"intel-rapl:0", "intel-rapl:1"} {
		base := "/sys/class/powercap/" + dom
		v, err1 := readInt(base + "/energy_uj")
		r, err2 := readInt(base + "/max_energy_range_uj")
		if err1 != nil || err2 != nil {
			v, r = -1, 0
		}
		e, rng = append(e, v), append(rng, r)
	}
	return
}

func cmdBench(args []string) {
	fs := flag.NewFlagSet("bench", flag.ExitOnError)
	out := fs.String("o", "bench-"+time.Now().Format("20060102-1504")+".csv", "CSV output")
	dutiesS := fs.String("duties", "10,15,20", "fan duties, %")
	capsS := fs.String("caps", "230,290,320", "power caps, W")
	hold := fs.Duration("hold", 3*time.Minute, "time per point")
	maxCPU := fs.Float64("max-cpu", 85, "abort at this CPU temperature (C)")
	maxExh := fs.Float64("max-exhaust", 60, "abort at this exhaust temperature (C)")
	fs.Parse(args)
	duties, err1 := intList(*dutiesS)
	caps, err2 := intList(*capsS)
	if err1 != nil || err2 != nil {
		log.Fatalf("bench: -duties / -caps: %v %v", err1, err2)
	}
	if fi, err := os.Stat(stateFile); err == nil && time.Since(fi.ModTime()) < 30*time.Second {
		log.Fatalf("bench: fanctl is running (%s written %s ago): sudo systemctl stop fanctl first",
			stateFile, time.Since(fi.ModTime()).Round(time.Second))
	}
	bmc, err := OpenBMC()
	if err != nil {
		log.Fatal(err)
	}
	sensors, err := bmc.Sensors()
	if err != nil {
		log.Fatalf("sensor repository: %v", err)
	}
	perc, perr := OpenPerc()
	var drives []Drive
	if perr == nil {
		drives = perc.Scan()
	}
	f, err := os.Create(*out)
	if err != nil {
		log.Fatal(err)
	}
	w := bufio.NewWriter(f)

	// the load: one AES worker per thread, in its own process group so all of it can be killed
	load := exec.Command("openssl", "speed", "-multi", strconv.Itoa(runtime.NumCPU()), "-seconds", "100000", "-evp", "aes-256-gcm")
	load.SysProcAttr = &syscall.SysProcAttr{Setpgid: true}
	cleanup := func(why string) {
		if load.Process != nil {
			syscall.Kill(-load.Process.Pid, syscall.SIGKILL)
		}
		w.Flush()
		f.Close()
		if err := fansToIDRAC(bmc); err != nil {
			log.Printf("%sfans -> iDRAC: %v", prErr, err)
		}
		_, capMax := capsForAuto()
		if err := bmc.SetPowerCap(capMax); err != nil {
			log.Printf("%spower cap %d W: %v", prErr, capMax, err)
		}
		log.Printf("bench %s: fans to the iDRAC, cap %d W, data in %s", why, capMax, *out)
	}
	sig := make(chan os.Signal, 1)
	signal.Notify(sig, syscall.SIGINT, syscall.SIGTERM)
	go func() {
		<-sig
		cleanup("interrupted")
		os.Exit(1)
	}()
	if err := load.Start(); err != nil {
		cleanup("failed")
		log.Fatalf("load: %v", err)
	}

	fmt.Fprintln(w, "unix,point,duty,cap,cpu1,cpu2,rapl0_w,rapl1_w,wall_w,inlet,exhaust,rpm_median,hdd_max")
	points := benchOrder(duties, caps)
	log.Printf("bench: %d points x %s, load on %d threads, writing %s", len(points), *hold, runtime.NumCPU(), *out)
	prevE, rng := raplJoules()
	prevT, hddMax, hddAt := time.Now(), 0.0, time.Time{}
	for i, p := range points {
		if err := bmc.SetPowerCap(p.cap); err != nil {
			log.Printf("%spoint %d: cap %d W: %v", prErr, i, p.cap, err)
		}
		if err := bmc.FanDuty(p.duty); err != nil {
			log.Printf("%spoint %d: duty %d %%: %v", prErr, i, p.duty, err)
		}
		log.Printf("bench point %d/%d: %d %% fans, %d W cap", i+1, len(points), p.duty, p.cap)
		for end := time.Now().Add(*hold); time.Now().Before(end); {
			time.Sleep(5 * time.Second)
			now := time.Now()
			e, _ := raplJoules()
			watts := make([]float64, 2)
			for s := range e {
				de := e[s] - prevE[s]
				if de < 0 {
					de += rng[s]
				}
				if e[s] >= 0 && prevE[s] >= 0 {
					watts[s] = float64(de) / 1e6 / now.Sub(prevT).Seconds()
				}
			}
			prevE, prevT = e, now
			cpus, _ := senseCoretemp()
			var wall, inlet, exhaust float64
			var rpm []float64
			for _, s := range sensors {
				v, ok, _, err := bmc.Reading(s)
				if err != nil || !ok {
					continue
				}
				switch k := sensorKey(s); {
				case s.Type == sensorFan:
					rpm = append(rpm, v)
				case k == "power":
					wall = v
				case k == "Inlet":
					inlet = v
				case k == "Exhaust":
					exhaust = v
				}
			}
			if perr == nil && now.Sub(hddAt) >= 30*time.Second { // drives are slow to read and slow to change
				hddMax, hddAt = 0, now
				for _, d := range drives {
					if t, ok := perc.Temp(d); ok && t > hddMax && t < 100 {
						hddMax = t
					}
				}
			}
			med := 0.0
			if len(rpm) > 0 {
				sort.Float64s(rpm)
				med = rpm[(len(rpm)-1)/2]
			}
			fmt.Fprintf(w, "%d,%d,%d,%d,%g,%g,%.1f,%.1f,%g,%g,%g,%g,%g\n", now.Unix(), i, p.duty, p.cap,
				cpus["CPU1"], cpus["CPU2"], watts[0], watts[1], wall, inlet, exhaust, med, hddMax)
			w.Flush()
			if hot := max(cpus["CPU1"], cpus["CPU2"]); hot >= *maxCPU || exhaust >= *maxExh {
				cleanup(fmt.Sprintf("aborted (CPU %g C, exhaust %g C)", hot, exhaust))
				os.Exit(1)
			}
		}
	}
	cleanup("done")
}
