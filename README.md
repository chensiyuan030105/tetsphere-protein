# tetsphere-protein


## How to Use

### 1. Data Preprocessing
To preprocess the protein data, you can run the following command, which uses the `data_processing_carbon.py` script. You need to specify the configuration file in YAML format that contains the details for the data processing step:

```bash
python data_preprocess/data_processing_carbon.py --config ./config/test/1A1U_C_test_16.yaml
python trainer.py --config ./config/test/1A1U_C_test_16.yaml
