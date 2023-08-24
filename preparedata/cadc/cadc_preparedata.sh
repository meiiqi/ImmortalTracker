path=$1
echo "Processing CADC_info, Dataset Path=${path}"
mkdir ./data
mkdir ./data/cadc
python preparedata/cadc/time_stamp.py --data_folder ${path}
python preparedata/cadc/time_stamp.py --data_folder ${path} --test
python preparedata/cadc/ego_info.py --data_folder ${path}
python preparedata/cadc/ego_info.py --data_folder ${path} --test