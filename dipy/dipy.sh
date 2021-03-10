# CSD Fit Truncated SVD for different percent truncations
python test.py --csd_fit 1 --read_data 0 --algorithm truncated_svd_by_value --percent_truncation 0.1
python test.py --csd_fit 1 --read_data 0 --algorithm truncated_svd_by_value --percent_truncation 0.2
python test.py --csd_fit 1 --read_data 0 --algorithm truncated_svd_by_value --percent_truncation 0.3
python test.py --csd_fit 1 --read_data 0 --algorithm truncated_svd_by_value --percent_truncation 0.4
python test.py --csd_fit 1 --read_data 0 --algorithm truncated_svd_by_value --percent_truncation 0.5
python test.py --csd_fit 1 --read_data 0 --algorithm truncated_svd_by_value --percent_truncation 0.6
python test.py --csd_fit 1 --read_data 0 --algorithm truncated_svd_by_value --percent_truncation 0.7
python test.py --csd_fit 1 --read_data 0 --algorithm truncated_svd_by_value --percent_truncation 0.8
python test.py --csd_fit 1 --read_data 0 --algorithm truncated_svd_by_value --percent_truncation 0.9
python test.py --csd_fit 1 --read_data 0 --algorithm truncated_svd_by_value --percent_truncation 1.0

#Error analysis for truncated SVD fit:
python test.py --csd_fit 0 --read_data 1 --algorithm truncated_svd_by_value --percent_truncation 0.1
python test.py --csd_fit 0 --read_data 1 --algorithm truncated_svd_by_value --percent_truncation 0.2
python test.py --csd_fit 0 --read_data 1 --algorithm truncated_svd_by_value --percent_truncation 0.3
python test.py --csd_fit 0 --read_data 1 --algorithm truncated_svd_by_value --percent_truncation 0.4
python test.py --csd_fit 0 --read_data 1 --algorithm truncated_svd_by_value --percent_truncation 0.5
python test.py --csd_fit 0 --read_data 1 --algorithm truncated_svd_by_value --percent_truncation 0.6
python test.py --csd_fit 0 --read_data 1 --algorithm truncated_svd_by_value --percent_truncation 0.7
python test.py --csd_fit 0 --read_data 1 --algorithm truncated_svd_by_value --percent_truncation 0.8
python test.py --csd_fit 0 --read_data 1 --algorithm truncated_svd_by_value --percent_truncation 0.9
python test.py --csd_fit 0 --read_data 1 --algorithm truncated_svd_by_value --percent_truncation 1.0

# python test.py --csd_fit 1 --read_data 0 --algorithm qr --percent_truncation 0.0
# python test.py --csd_fit 1 --read_data 0 --algorithm svd --percent_truncation 0.0
# python test.py --csd_fit 0 --read_data 1 --algorithm qr --percent_truncation 0.0
# python test.py --csd_fit 0 --read_data 1 --algorithm svd --percent_truncation 0.0

