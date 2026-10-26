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

### Every cell

| Mode | Input | Case | m | Algorithm | C++ ns | Rust ns | Rust / C++ |
|---|---|---|---:|---|---:|---:|---:|
| adversarial | ab | absent | 16 | Guardzilla | 38049.0 | 20591.0 | 0.54 |
| adversarial | ab | absent | 16 | NeedleHammer | 42327.0 | 34646.0 | 0.82 |
| adversarial | ab | absent | 32 | Guardzilla | 38135.0 | 20770.0 | 0.54 |
| adversarial | ab | absent | 32 | NeedleHammer | 42374.0 | 34779.0 | 0.82 |
| adversarial | ab | absent | 64 | Guardzilla | 37079.0 | 20123.0 | 0.54 |
| adversarial | ab | absent | 64 | NeedleHammer | 41193.0 | 33452.0 | 0.81 |
| adversarial | ab | absent | 128 | Guardzilla | 36602.0 | 19313.0 | 0.53 |
| adversarial | ab | absent | 128 | NeedleHammer | 36362.0 | 31503.0 | 0.87 |
| adversarial | ab | absent | 256 | Guardzilla | 36455.0 | 19120.0 | 0.52 |
| adversarial | ab | absent | 256 | NeedleHammer | 36599.0 | 32159.0 | 0.88 |
| adversarial | ab | absent | 512 | Guardzilla | 35870.0 | 18909.0 | 0.53 |
| adversarial | ab | absent | 512 | NeedleHammer | 35412.0 | 30488.0 | 0.86 |
| adversarial | ab | absent | 1024 | Guardzilla | 35139.0 | 19563.0 | 0.56 |
| adversarial | ab | absent | 1024 | NeedleHammer | 34873.0 | 30805.0 | 0.88 |
| adversarial | ab | absent | 2048 | Guardzilla | 35160.0 | 19886.0 | 0.57 |
| adversarial | ab | absent | 2048 | NeedleHammer | 37213.0 | 32200.0 | 0.87 |
| adversarial | ab | absent | 4096 | Guardzilla | 35625.0 | 20967.0 | 0.59 |
| adversarial | ab | absent | 4096 | NeedleHammer | 34875.0 | 33606.0 | 0.96 |
| adversarial | block | absent | 16 | Guardzilla | 659722.0 | 820680.0 | 1.24 |
| adversarial | block | absent | 16 | NeedleHammer | 689519.0 | 1058162.0 | 1.53 |
| adversarial | block | absent | 32 | Guardzilla | 710340.0 | 876952.0 | 1.23 |
| adversarial | block | absent | 32 | NeedleHammer | 705364.0 | 1050282.0 | 1.49 |
| adversarial | block | absent | 64 | Guardzilla | 755432.0 | 918032.0 | 1.22 |
| adversarial | block | absent | 64 | NeedleHammer | 794011.0 | 1113250.0 | 1.40 |
| adversarial | block | absent | 128 | Guardzilla | 108696.0 | 100699.0 | 0.93 |
| adversarial | block | absent | 128 | NeedleHammer | 108488.0 | 103555.0 | 0.95 |
| adversarial | block | absent | 256 | Guardzilla | 60990.0 | 58434.0 | 0.96 |
| adversarial | block | absent | 256 | NeedleHammer | 61993.0 | 59511.0 | 0.96 |
| adversarial | block | absent | 512 | Guardzilla | 40024.0 | 37774.0 | 0.94 |
| adversarial | block | absent | 512 | NeedleHammer | 39427.0 | 39108.0 | 0.99 |
| adversarial | block | absent | 1024 | Guardzilla | 32482.0 | 29704.0 | 0.91 |
| adversarial | block | absent | 1024 | NeedleHammer | 31213.0 | 30891.0 | 0.99 |
| adversarial | block | absent | 2048 | Guardzilla | 32849.0 | 31110.0 | 0.95 |
| adversarial | block | absent | 2048 | NeedleHammer | 30854.0 | 32350.0 | 1.05 |
| adversarial | block | absent | 4096 | Guardzilla | 37786.0 | 37806.0 | 1.00 |
| adversarial | block | absent | 4096 | NeedleHammer | 36505.0 | 36879.0 | 1.01 |
| adversarial | mid | absent | 16 | Guardzilla | 40543.0 | 25260.0 | 0.62 |
| adversarial | mid | absent | 16 | NeedleHammer | 42748.0 | 39160.0 | 0.92 |
| adversarial | mid | absent | 32 | Guardzilla | 40586.0 | 24509.0 | 0.60 |
| adversarial | mid | absent | 32 | NeedleHammer | 43942.0 | 38062.0 | 0.87 |
| adversarial | mid | absent | 64 | Guardzilla | 40727.0 | 23708.0 | 0.58 |
| adversarial | mid | absent | 64 | NeedleHammer | 43426.0 | 37875.0 | 0.87 |
| adversarial | mid | absent | 128 | Guardzilla | 38041.0 | 17548.0 | 0.46 |
| adversarial | mid | absent | 128 | NeedleHammer | 40502.0 | 35318.0 | 0.87 |
| adversarial | mid | absent | 256 | Guardzilla | 37979.0 | 17761.0 | 0.47 |
| adversarial | mid | absent | 256 | NeedleHammer | 41557.0 | 33434.0 | 0.80 |
| adversarial | mid | absent | 512 | Guardzilla | 38653.0 | 20227.0 | 0.52 |
| adversarial | mid | absent | 512 | NeedleHammer | 41976.0 | 35598.0 | 0.85 |
| adversarial | mid | absent | 1024 | Guardzilla | 37888.0 | 20648.0 | 0.54 |
| adversarial | mid | absent | 1024 | NeedleHammer | 41033.0 | 34657.0 | 0.84 |
| adversarial | mid | absent | 2048 | Guardzilla | 38303.0 | 20839.0 | 0.54 |
| adversarial | mid | absent | 2048 | NeedleHammer | 42803.0 | 34524.0 | 0.81 |
| adversarial | mid | absent | 4096 | Guardzilla | 39878.0 | 22600.0 | 0.57 |
| adversarial | mid | absent | 4096 | NeedleHammer | 44763.0 | 35840.0 | 0.80 |
| adversarial | near | absent | 16 | Guardzilla | 40703.0 | 25680.0 | 0.63 |
| adversarial | near | absent | 16 | NeedleHammer | 41719.0 | 42449.0 | 1.02 |
| adversarial | near | absent | 32 | Guardzilla | 40708.0 | 24112.0 | 0.59 |
| adversarial | near | absent | 32 | NeedleHammer | 42855.0 | 39816.0 | 0.93 |
| adversarial | near | absent | 64 | Guardzilla | 40599.0 | 23867.0 | 0.59 |
| adversarial | near | absent | 64 | NeedleHammer | 42838.0 | 40197.0 | 0.94 |
| adversarial | near | absent | 128 | Guardzilla | 45272.0 | 32528.0 | 0.72 |
| adversarial | near | absent | 128 | NeedleHammer | 44598.0 | 46939.0 | 1.05 |
| adversarial | near | absent | 256 | Guardzilla | 40688.0 | 24148.0 | 0.59 |
| adversarial | near | absent | 256 | NeedleHammer | 43825.0 | 38096.0 | 0.87 |
| adversarial | near | absent | 512 | Guardzilla | 47108.0 | 34642.0 | 0.74 |
| adversarial | near | absent | 512 | NeedleHammer | 49310.0 | 53192.0 | 1.08 |
| adversarial | near | absent | 1024 | Guardzilla | 61524.0 | 67966.0 | 1.10 |
| adversarial | near | absent | 1024 | NeedleHammer | 65951.0 | 77535.0 | 1.18 |
| adversarial | near | absent | 2048 | Guardzilla | 98682.0 | 121024.0 | 1.23 |
| adversarial | near | absent | 2048 | NeedleHammer | 100241.0 | 125091.0 | 1.25 |
| adversarial | near | absent | 4096 | Guardzilla | 35983.0 | 39183.0 | 1.09 |
| adversarial | near | absent | 4096 | NeedleHammer | 42238.0 | 40825.0 | 0.97 |
| adversarial | tail | absent | 16 | Guardzilla | 40693.0 | 24242.0 | 0.60 |
| adversarial | tail | absent | 16 | NeedleHammer | 41548.0 | 39440.0 | 0.95 |
| adversarial | tail | absent | 32 | Guardzilla | 40737.0 | 24108.0 | 0.59 |
| adversarial | tail | absent | 32 | NeedleHammer | 43997.0 | 39354.0 | 0.89 |
| adversarial | tail | absent | 64 | Guardzilla | 41118.0 | 24174.0 | 0.59 |
| adversarial | tail | absent | 64 | NeedleHammer | 43268.0 | 40202.0 | 0.93 |
| adversarial | tail | absent | 128 | Guardzilla | 41208.0 | 25438.0 | 0.62 |
| adversarial | tail | absent | 128 | NeedleHammer | 42034.0 | 40328.0 | 0.96 |
| adversarial | tail | absent | 256 | Guardzilla | 40122.0 | 25123.0 | 0.63 |
| adversarial | tail | absent | 256 | NeedleHammer | 45616.0 | 40370.0 | 0.88 |
| adversarial | tail | absent | 512 | Guardzilla | 41352.0 | 26530.0 | 0.64 |
| adversarial | tail | absent | 512 | NeedleHammer | 44647.0 | 38208.0 | 0.86 |
| adversarial | tail | absent | 1024 | Guardzilla | 41444.0 | 25080.0 | 0.61 |
| adversarial | tail | absent | 1024 | NeedleHammer | 44576.0 | 38871.0 | 0.87 |
| adversarial | tail | absent | 2048 | Guardzilla | 42980.0 | 25750.0 | 0.60 |
| adversarial | tail | absent | 2048 | NeedleHammer | 45468.0 | 39372.0 | 0.87 |
| adversarial | tail | absent | 4096 | Guardzilla | 45290.0 | 28393.0 | 0.63 |
| adversarial | tail | absent | 4096 | NeedleHammer | 46815.0 | 43358.0 | 0.93 |
| corpora | base64 | absent | 8 | Guardzilla | 37116.9 | 17506.8 | 0.47 |
| corpora | base64 | absent | 8 | NeedleHammer | 41816.3 | 33673.1 | 0.81 |
| corpora | base64 | absent | 32 | Guardzilla | 37019.8 | 17666.3 | 0.48 |
| corpora | base64 | absent | 32 | NeedleHammer | 41221.2 | 34405.6 | 0.83 |
| corpora | base64 | absent | 256 | Guardzilla | 35523.4 | 16433.5 | 0.46 |
| corpora | base64 | absent | 256 | NeedleHammer | 36114.2 | 31425.9 | 0.87 |
| corpora | base64 | absent | 2048 | Guardzilla | 41490.8 | 19101.0 | 0.46 |
| corpora | base64 | absent | 2048 | NeedleHammer | 36860.6 | 30615.8 | 0.83 |
| corpora | base64 | present | 8 | Guardzilla | 17238.9 | 8217.8 | 0.48 |
| corpora | base64 | present | 8 | NeedleHammer | 19292.3 | 16515.9 | 0.86 |
| corpora | base64 | present | 32 | Guardzilla | 19477.6 | 9340.8 | 0.48 |
| corpora | base64 | present | 32 | NeedleHammer | 21741.1 | 17335.8 | 0.80 |
| corpora | base64 | present | 256 | Guardzilla | 17607.0 | 8415.0 | 0.48 |
| corpora | base64 | present | 256 | NeedleHammer | 18454.8 | 14947.9 | 0.81 |
| corpora | base64 | present | 2048 | Guardzilla | 18946.2 | 8689.0 | 0.46 |
| corpora | base64 | present | 2048 | NeedleHammer | 16762.2 | 14074.0 | 0.84 |
| corpora | chinese | absent | 8 | Guardzilla | 37695.8 | 17402.6 | 0.46 |
| corpora | chinese | absent | 8 | NeedleHammer | 41231.6 | 35391.4 | 0.86 |
| corpora | chinese | absent | 32 | Guardzilla | 40367.3 | 22387.7 | 0.55 |
| corpora | chinese | absent | 32 | NeedleHammer | 43238.2 | 40662.0 | 0.94 |
| corpora | chinese | absent | 256 | Guardzilla | 39703.2 | 22342.3 | 0.56 |
| corpora | chinese | absent | 256 | NeedleHammer | 45097.4 | 39282.2 | 0.87 |
| corpora | chinese | absent | 2048 | Guardzilla | 42795.6 | 19769.9 | 0.46 |
| corpora | chinese | absent | 2048 | NeedleHammer | 39137.8 | 34135.7 | 0.87 |
| corpora | chinese | present | 8 | Guardzilla | 4043.1 | 1950.5 | 0.48 |
| corpora | chinese | present | 8 | NeedleHammer | 4454.3 | 3683.8 | 0.83 |
| corpora | chinese | present | 32 | Guardzilla | 4746.1 | 2675.7 | 0.56 |
| corpora | chinese | present | 32 | NeedleHammer | 5198.0 | 4745.4 | 0.91 |
| corpora | chinese | present | 256 | Guardzilla | 4829.8 | 2855.6 | 0.59 |
| corpora | chinese | present | 256 | NeedleHammer | 5473.8 | 4650.4 | 0.85 |
| corpora | chinese | present | 2048 | Guardzilla | 5246.4 | 2558.4 | 0.49 |
| corpora | chinese | present | 2048 | NeedleHammer | 4937.2 | 4281.8 | 0.87 |
| corpora | dna | absent | 32 | Guardzilla | 183804.5 | 159862.6 | 0.87 |
| corpora | dna | absent | 32 | NeedleHammer | 49682.8 | 50650.0 | 1.02 |
| corpora | dna | absent | 256 | Guardzilla | 197668.0 | 167470.0 | 0.85 |
| corpora | dna | absent | 256 | NeedleHammer | 45826.2 | 47754.9 | 1.04 |
| corpora | dna | absent | 2048 | Guardzilla | 195035.9 | 173495.8 | 0.89 |
| corpora | dna | absent | 2048 | NeedleHammer | 48680.8 | 47261.1 | 0.97 |
| corpora | dna | present | 8 | Guardzilla | 9303.9 | 7966.2 | 0.86 |
| corpora | dna | present | 8 | NeedleHammer | 3650.2 | 3918.8 | 1.07 |
| corpora | dna | present | 32 | Guardzilla | 94575.8 | 82497.6 | 0.87 |
| corpora | dna | present | 32 | NeedleHammer | 25003.4 | 25535.0 | 1.02 |
| corpora | dna | present | 256 | Guardzilla | 92391.9 | 84457.7 | 0.91 |
| corpora | dna | present | 256 | NeedleHammer | 23566.5 | 24563.3 | 1.04 |
| corpora | dna | present | 2048 | Guardzilla | 83860.4 | 77499.4 | 0.92 |
| corpora | dna | present | 2048 | NeedleHammer | 22739.7 | 23042.5 | 1.01 |
| corpora | english | absent | 8 | Guardzilla | 38527.7 | 19358.8 | 0.50 |
| corpora | english | absent | 8 | NeedleHammer | 42606.8 | 35843.7 | 0.84 |
| corpora | english | absent | 32 | Guardzilla | 38915.4 | 19283.8 | 0.50 |
| corpora | english | absent | 32 | NeedleHammer | 42949.9 | 35563.8 | 0.83 |
| corpora | english | absent | 256 | Guardzilla | 37481.0 | 18654.0 | 0.50 |
| corpora | english | absent | 256 | NeedleHammer | 38540.4 | 34655.9 | 0.90 |
| corpora | english | absent | 2048 | Guardzilla | 43212.2 | 21183.2 | 0.49 |
| corpora | english | absent | 2048 | NeedleHammer | 38696.1 | 32841.5 | 0.85 |
| corpora | english | present | 8 | Guardzilla | 2006.4 | 1039.4 | 0.52 |
| corpora | english | present | 8 | NeedleHammer | 2165.5 | 1788.7 | 0.83 |
| corpora | english | present | 32 | Guardzilla | 2569.4 | 1305.5 | 0.51 |
| corpora | english | present | 32 | NeedleHammer | 2718.5 | 2296.7 | 0.84 |
| corpora | english | present | 256 | Guardzilla | 2277.6 | 1157.1 | 0.51 |
| corpora | english | present | 256 | NeedleHammer | 2381.7 | 1966.2 | 0.83 |
| corpora | english | present | 2048 | Guardzilla | 2910.0 | 1473.3 | 0.51 |
| corpora | english | present | 2048 | NeedleHammer | 2590.5 | 2168.6 | 0.84 |
| corpora | influenza | absent | 8 | Guardzilla | 161722.9 | 137881.9 | 0.85 |
| corpora | influenza | absent | 8 | NeedleHammer | 46843.9 | 48030.9 | 1.03 |
| corpora | influenza | absent | 32 | Guardzilla | 163739.6 | 138702.6 | 0.85 |
| corpora | influenza | absent | 32 | NeedleHammer | 45984.1 | 46756.2 | 1.02 |
| corpora | influenza | absent | 256 | Guardzilla | 161363.0 | 143166.1 | 0.89 |
| corpora | influenza | absent | 256 | NeedleHammer | 45741.7 | 46031.6 | 1.01 |
| corpora | influenza | absent | 2048 | Guardzilla | 165377.1 | 145580.6 | 0.88 |
| corpora | influenza | absent | 2048 | NeedleHammer | 49577.1 | 47617.1 | 0.96 |
| corpora | influenza | present | 8 | Guardzilla | 28597.6 | 24190.9 | 0.85 |
| corpora | influenza | present | 8 | NeedleHammer | 10424.5 | 11335.7 | 1.09 |
| corpora | influenza | present | 32 | Guardzilla | 75152.7 | 63456.2 | 0.84 |
| corpora | influenza | present | 32 | NeedleHammer | 22616.6 | 24072.4 | 1.06 |
| corpora | influenza | present | 256 | Guardzilla | 76790.1 | 67788.5 | 0.88 |
| corpora | influenza | present | 256 | NeedleHammer | 23601.2 | 25322.9 | 1.07 |
| corpora | influenza | present | 2048 | Guardzilla | 73153.0 | 66159.1 | 0.90 |
| corpora | influenza | present | 2048 | NeedleHammer | 23360.9 | 24665.7 | 1.06 |
| corpora | json | absent | 8 | Guardzilla | 38319.2 | 18302.5 | 0.48 |
| corpora | json | absent | 8 | NeedleHammer | 41555.1 | 34612.4 | 0.83 |
| corpora | json | absent | 32 | Guardzilla | 39052.7 | 20053.2 | 0.51 |
| corpora | json | absent | 32 | NeedleHammer | 42543.6 | 35824.7 | 0.84 |
| corpora | json | absent | 256 | Guardzilla | 36030.7 | 17629.0 | 0.49 |
| corpora | json | absent | 256 | NeedleHammer | 38260.6 | 33088.8 | 0.86 |
| corpora | json | absent | 2048 | Guardzilla | 43065.9 | 20459.2 | 0.48 |
| corpora | json | absent | 2048 | NeedleHammer | 37514.8 | 32742.5 | 0.87 |
| corpora | json | present | 8 | Guardzilla | 11544.5 | 5997.3 | 0.52 |
| corpora | json | present | 8 | NeedleHammer | 12024.9 | 10633.0 | 0.88 |
| corpora | json | present | 32 | Guardzilla | 20620.9 | 10522.2 | 0.51 |
| corpora | json | present | 32 | NeedleHammer | 22384.7 | 19168.0 | 0.86 |
| corpora | json | present | 256 | Guardzilla | 18414.7 | 9016.8 | 0.49 |
| corpora | json | present | 256 | NeedleHammer | 19061.8 | 16759.7 | 0.88 |
| corpora | json | present | 2048 | Guardzilla | 19683.3 | 9460.4 | 0.48 |
| corpora | json | present | 2048 | NeedleHammer | 17123.6 | 15108.1 | 0.88 |
| corpora | leaders | absent | 8 | Guardzilla | 37485.6 | 17856.9 | 0.48 |
| corpora | leaders | absent | 8 | NeedleHammer | 41911.3 | 36235.0 | 0.86 |
| corpora | leaders | absent | 32 | Guardzilla | 58667.3 | 39405.7 | 0.67 |
| corpora | leaders | absent | 32 | NeedleHammer | 46982.2 | 43278.6 | 0.92 |
| corpora | leaders | absent | 256 | Guardzilla | 43845.9 | 26501.8 | 0.60 |
| corpora | leaders | absent | 256 | NeedleHammer | 46677.9 | 42739.5 | 0.92 |
| corpora | leaders | absent | 2048 | Guardzilla | 46192.9 | 24718.4 | 0.54 |
| corpora | leaders | absent | 2048 | NeedleHammer | 41363.1 | 36809.6 | 0.89 |
| corpora | leaders | present | 8 | Guardzilla | 2482.5 | 1193.1 | 0.48 |
| corpora | leaders | present | 8 | NeedleHammer | 2746.8 | 2392.1 | 0.87 |
| corpora | leaders | present | 32 | Guardzilla | 5243.2 | 2985.8 | 0.57 |
| corpora | leaders | present | 32 | NeedleHammer | 4901.7 | 4496.9 | 0.92 |
| corpora | leaders | present | 256 | Guardzilla | 12831.7 | 7721.5 | 0.60 |
| corpora | leaders | present | 256 | NeedleHammer | 13493.5 | 12710.1 | 0.94 |
| corpora | leaders | present | 2048 | Guardzilla | 16140.7 | 9405.1 | 0.58 |
| corpora | leaders | present | 2048 | NeedleHammer | 14960.9 | 13250.3 | 0.89 |
| corpora | log | absent | 8 | Guardzilla | 38923.8 | 19684.4 | 0.51 |
| corpora | log | absent | 8 | NeedleHammer | 42623.2 | 37830.1 | 0.89 |
| corpora | log | absent | 32 | Guardzilla | 39560.2 | 19082.2 | 0.48 |
| corpora | log | absent | 32 | NeedleHammer | 41460.4 | 36721.6 | 0.89 |
| corpora | log | absent | 256 | Guardzilla | 37289.6 | 18341.2 | 0.49 |
| corpora | log | absent | 256 | NeedleHammer | 38005.6 | 35207.0 | 0.93 |
| corpora | log | absent | 2048 | Guardzilla | 42479.6 | 19982.7 | 0.47 |
| corpora | log | absent | 2048 | NeedleHammer | 37279.5 | 32174.4 | 0.86 |
| corpora | log | present | 8 | Guardzilla | 3091.8 | 1878.6 | 0.61 |
| corpora | log | present | 8 | NeedleHammer | 3121.4 | 2854.4 | 0.91 |
| corpora | log | present | 32 | Guardzilla | 26639.6 | 18465.3 | 0.69 |
| corpora | log | present | 32 | NeedleHammer | 21329.7 | 21039.4 | 0.99 |
| corpora | log | present | 256 | Guardzilla | 19160.8 | 10058.1 | 0.52 |
| corpora | log | present | 256 | NeedleHammer | 20218.6 | 17410.7 | 0.86 |
| corpora | log | present | 2048 | Guardzilla | 19678.5 | 9786.1 | 0.50 |
| corpora | log | present | 2048 | NeedleHammer | 18214.6 | 14593.8 | 0.80 |
| corpora | protein | absent | 8 | Guardzilla | 38778.3 | 19329.6 | 0.50 |
| corpora | protein | absent | 8 | NeedleHammer | 42208.0 | 36118.8 | 0.86 |
| corpora | protein | absent | 32 | Guardzilla | 40080.5 | 19508.8 | 0.49 |
| corpora | protein | absent | 32 | NeedleHammer | 42455.4 | 36333.4 | 0.86 |
| corpora | protein | absent | 256 | Guardzilla | 36983.9 | 18422.8 | 0.50 |
| corpora | protein | absent | 256 | NeedleHammer | 37939.2 | 31997.9 | 0.84 |
| corpora | protein | absent | 2048 | Guardzilla | 43751.4 | 22243.0 | 0.51 |
| corpora | protein | absent | 2048 | NeedleHammer | 39011.4 | 32136.7 | 0.82 |
| corpora | protein | present | 8 | Guardzilla | 18222.2 | 9032.1 | 0.50 |
| corpora | protein | present | 8 | NeedleHammer | 19890.0 | 17042.5 | 0.86 |
| corpora | protein | present | 32 | Guardzilla | 20438.0 | 10161.1 | 0.50 |
| corpora | protein | present | 32 | NeedleHammer | 22032.6 | 19202.2 | 0.87 |
| corpora | protein | present | 256 | Guardzilla | 18718.7 | 9249.2 | 0.49 |
| corpora | protein | present | 256 | NeedleHammer | 19449.9 | 16112.6 | 0.83 |
| corpora | protein | present | 2048 | Guardzilla | 19710.8 | 9839.2 | 0.50 |
| corpora | protein | present | 2048 | NeedleHammer | 17939.4 | 14406.1 | 0.80 |
| corpora | replog | absent | 8 | Guardzilla | 37600.8 | 17432.8 | 0.46 |
| corpora | replog | absent | 8 | NeedleHammer | 41263.3 | 35388.4 | 0.86 |
| corpora | replog | absent | 32 | Guardzilla | 37249.7 | 17404.8 | 0.47 |
| corpora | replog | absent | 32 | NeedleHammer | 41274.3 | 35927.2 | 0.87 |
| corpora | replog | absent | 256 | Guardzilla | 35894.1 | 16705.7 | 0.47 |
| corpora | replog | absent | 256 | NeedleHammer | 37586.1 | 31920.5 | 0.85 |
| corpora | replog | absent | 2048 | Guardzilla | 43329.5 | 20968.2 | 0.48 |
| corpora | replog | absent | 2048 | NeedleHammer | 39415.1 | 32318.8 | 0.82 |
| corpora | replog | present | 8 | Guardzilla | 2565.6 | 1385.2 | 0.54 |
| corpora | replog | present | 8 | NeedleHammer | 2725.7 | 2491.2 | 0.91 |
| corpora | replog | present | 32 | Guardzilla | 13460.2 | 11378.0 | 0.85 |
| corpora | replog | present | 32 | NeedleHammer | 14030.0 | 13835.4 | 0.99 |
| corpora | replog | present | 256 | Guardzilla | 22189.3 | 21165.9 | 0.95 |
| corpora | replog | present | 256 | NeedleHammer | 23220.8 | 22641.4 | 0.98 |
| corpora | replog | present | 2048 | Guardzilla | 21290.2 | 19504.0 | 0.92 |
| corpora | replog | present | 2048 | NeedleHammer | 21730.4 | 20666.7 | 0.95 |
| corpora | russian | absent | 8 | Guardzilla | 46890.9 | 27259.0 | 0.58 |
| corpora | russian | absent | 8 | NeedleHammer | 45132.8 | 41211.8 | 0.91 |
| corpora | russian | absent | 32 | Guardzilla | 38883.7 | 21665.4 | 0.56 |
| corpora | russian | absent | 32 | NeedleHammer | 42823.6 | 39255.3 | 0.92 |
| corpora | russian | absent | 256 | Guardzilla | 37441.2 | 18665.0 | 0.50 |
| corpora | russian | absent | 256 | NeedleHammer | 41515.7 | 35505.2 | 0.86 |
| corpora | russian | absent | 2048 | Guardzilla | 42835.2 | 23134.0 | 0.54 |
| corpora | russian | absent | 2048 | NeedleHammer | 40907.3 | 35949.2 | 0.88 |
| corpora | russian | present | 8 | Guardzilla | 1213.8 | 724.4 | 0.60 |
| corpora | russian | present | 8 | NeedleHammer | 1159.7 | 1042.0 | 0.90 |
| corpora | russian | present | 32 | Guardzilla | 1496.1 | 783.8 | 0.52 |
| corpora | russian | present | 32 | NeedleHammer | 1601.1 | 1393.7 | 0.87 |
| corpora | russian | present | 256 | Guardzilla | 1490.7 | 740.2 | 0.50 |
| corpora | russian | present | 256 | NeedleHammer | 1620.0 | 1265.0 | 0.78 |
| corpora | russian | present | 2048 | Guardzilla | 1741.7 | 917.3 | 0.53 |
| corpora | russian | present | 2048 | NeedleHammer | 1642.3 | 1384.2 | 0.84 |
| corpora | source | absent | 8 | Guardzilla | 37570.3 | 18060.4 | 0.48 |
| corpora | source | absent | 8 | NeedleHammer | 41397.4 | 36299.3 | 0.88 |
| corpora | source | absent | 32 | Guardzilla | 37793.9 | 18052.5 | 0.48 |
| corpora | source | absent | 32 | NeedleHammer | 41916.8 | 36908.2 | 0.88 |
| corpora | source | absent | 256 | Guardzilla | 36204.0 | 17537.2 | 0.48 |
| corpora | source | absent | 256 | NeedleHammer | 38154.8 | 33417.2 | 0.88 |
| corpora | source | absent | 2048 | Guardzilla | 42092.8 | 19832.0 | 0.47 |
| corpora | source | absent | 2048 | NeedleHammer | 36909.6 | 31985.5 | 0.87 |
| corpora | source | present | 8 | Guardzilla | 2879.6 | 1460.2 | 0.51 |
| corpora | source | present | 8 | NeedleHammer | 3059.4 | 2588.1 | 0.85 |
| corpora | source | present | 32 | Guardzilla | 4610.3 | 2289.0 | 0.50 |
| corpora | source | present | 32 | NeedleHammer | 5047.8 | 4485.4 | 0.89 |
| corpora | source | present | 256 | Guardzilla | 4428.3 | 2166.1 | 0.49 |
| corpora | source | present | 256 | NeedleHammer | 4527.2 | 4002.4 | 0.88 |
| corpora | source | present | 2048 | Guardzilla | 5021.1 | 2491.5 | 0.50 |
| corpora | source | present | 2048 | NeedleHammer | 4438.1 | 3854.7 | 0.87 |
