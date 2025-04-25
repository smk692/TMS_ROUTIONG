# TMS Routing Project Skeleton

## Directory Structure

```
tms-routing/
├── data/                       
│   └── sample_orders.csv
├── notebooks/                  
│   ├── 01_fetch_coordinates.ipynb
│   ├── 02_distance_matrix.ipynb
│   └── 03_tsp_solver.ipynb
├── src/                        
│   ├── db/                     
│   │   └── maria_client.py     
│   ├── model/                  
│   │   └── distance.py         
│   ├── algorithm/              
│   │   ├── tsp.py              
│   │   └── vrp.py              
│   └── config.py               
├── requirements.txt
└── README.md
```

## Setup

   pip install -e .
   
1. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run Jupyter notebooks in the `notebooks/` directory for experiments.
