# python testing.py --name fine_tuning_bubbles_max0.1 --max_depth 0.1 --masked
# python testing.py --name fine_tuning_bubbles_max0.12_no_mask_not_zero_train_tools_improved_data --max_depth 0.12
python testing.py --name fine_tuning_bubbles_max0.12_no_mask_not_zero_train_tools_improved_data_train_split --max_depth 0.12
python testing.py --name fine_tuning_bubbles_max0.12_no_mask_not_zero_train_tools_improved_data_scaled --max_depth 0.12 --scale 1000
python testing.py --name fine_tuning_bubbles_max0.20_no_mask_not_zero_train_tools_improved_data_scaled --max_depth 0.20 --scale 1000