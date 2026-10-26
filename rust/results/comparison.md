### Rust / C++ time ratio (below 1: Rust faster)

| Algorithm | Workload | Cells | Rust / C++ time (geomean) | Best | Worst |
|---|---|---:|---:|---:|---:|
| Guardzilla | corpora | 95 | 0.564 | 0.46 | 0.95 |
| Guardzilla | adversarial | 45 | 0.680 | 0.46 | 1.24 |
| NeedleHammer | corpora | 95 | 0.892 | 0.78 | 1.09 |
| NeedleHammer | adversarial | 45 | 0.948 | 0.80 | 1.53 |

### Throughput, absent needles (GB/s, geomean over m = 8, 32, 256, 2048)

| Corpus | C++ Guardzilla | Rust Guardzilla | C++ NeedleHammer | Rust NeedleHammer |
|---|---:|---:|---:|---:|
| english | 26.6 | 53.5 | 25.8 | 30.2 |
| log | 26.5 | 54.4 | 26.4 | 29.6 |
| json | 26.9 | 55.0 | 26.3 | 30.8 |
| base64 | 27.8 | 59.4 | 26.9 | 32.3 |
| protein | 26.3 | 52.9 | 26.0 | 30.8 |
| dna | 5.5 | 6.3 | 21.8 | 21.6 |
| russian | 25.4 | 46.7 | 24.6 | 27.7 |
| chinese | 26.1 | 51.5 | 24.9 | 28.1 |
| source | 27.3 | 57.1 | 26.5 | 30.3 |
| replog | 27.3 | 58.1 | 26.3 | 31.0 |
| influenza | 6.4 | 7.4 | 22.3 | 22.3 |
| leaders | 22.8 | 40.2 | 23.7 | 26.5 |

### Throughput, present needles (GB/s of haystack up to the match)

| Corpus | C++ Guardzilla | Rust Guardzilla | C++ NeedleHammer | Rust NeedleHammer |
|---|---:|---:|---:|---:|
| english | 26.3 | 51.5 | 25.9 | 31.0 |
| log | 21.8 | 37.8 | 23.1 | 26.0 |
| json | 26.0 | 52.0 | 25.9 | 29.5 |
| base64 | 27.5 | 58.2 | 26.5 | 32.2 |
| protein | 26.2 | 52.7 | 25.5 | 30.3 |
| dna | 5.7 | 6.4 | 19.7 | 19.0 |
| russian | 25.7 | 48.0 | 25.4 | 29.9 |
| chinese | 26.4 | 49.9 | 24.8 | 28.7 |
| source | 26.7 | 53.7 | 26.4 | 30.3 |
| replog | 24.2 | 30.5 | 23.2 | 24.3 |
| influenza | 6.5 | 7.4 | 20.0 | 18.7 |
| leaders | 23.6 | 42.4 | 23.6 | 26.1 |
