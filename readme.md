

## TGFD Correction

This repo contains scripts and data to run **TGFD correction experiments** (Earth Mover’s Distance–based repair) on temporal graph data.

It is meant to be used together with the main TGFD project:

- TGFD (mining + violation detection): https://github.com/TGFD-Project/TGFD

TGFD gives you **temporal snapshots** and **violation files**. This repo runs **correction** on those violations.

## Data in this repo


- `snap10/`, `snap25/`, `snap40/`, `snap60/`  
  Example **experiment folders**. Each one is a place to put:
  - TGFD rule file(s)
  - Violation file(s) produced by the TGFD project
  - Any simple config/notes for that experiment

- `results/`  
  Output files and summaries after you run an experiment.

- `run-snap-emd.sh`  
  Main shell script that runs the correction pipeline for a given experiment folder.

---

## How to run (step by step)

### 1. Generate data with TGFD

Do this in the **TGFD** repo (not here):

1. Follow the instructions from the TGFD README to:
   - download a dataset (e.g., IMDB), and  
   - run mining + error detection.

2. After that, you should have:
   - a set of **snapshot files** (temporal graph), and  
   - one or more **violation files** for chosen TGFDs.

### 2. Clone this repo
```bash
git clone https://github.com/tgfd-correction/tgfd-correction.git
cd tgfd-correction

```

### 3. Put the data in place

1.  Copy or symlink your **snapshot files** into `snapshots/`:
    
    ```bash
    ls snapshots/
    # should list your temporal graph snapshots
    
    ```
    
2.  Choose one experiment folder (for example, `snap10/`) and copy or symlink into it:
        
    -   the matching violation file(s)
        
    
    Example layout:
    
    ```text
    tgfd-correction/
      snapshots/
        snapshot_1
        snapshot_2
        ...
      snap10/
        my_violations.txt
    
    ```
    

### 4. Run the correction script

From the repo root:

1.  Make the script executable (only once):
    
    ```bash
    chmod +x run-snap-emd.sh
    
    ```
    
2.  Run an experiment folder, e.g. `snap10`:
    
    ```bash
    ./run-snap-emd.sh snap10
    
    ```
    
    You can also run the others similarly:
    
    ```bash
    ./run-snap-emd.sh snap25
    ./run-snap-emd.sh snap40
    ./run-snap-emd.sh snap60
    
    ```
    

The script will:

-   read snapshots from `snapshots/`
    
-   read TGFD / violation files from the chosen folder
    
-   run the EMD-based correction
    
-   write outputs to `results/` and/or the experiment folder
    


