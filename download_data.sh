mkdir -p data
cd data

####################### Download StatQA data
mkdir -p StatQA
cd StatQA
wget -nc https://raw.githubusercontent.com/HKUSTDial/StatQA/refs/heads/main/StatQA/mini-StatQA.json
wget -nc https://raw.githubusercontent.com/HKUSTDial/StatQA/refs/heads/main/StatQA/mini-StatQA.csv
wget -nc https://raw.githubusercontent.com/HKUSTDial/StatQA/refs/heads/main/Data/Metadata/Dataset%20metadata.csv
mv Dataset\ metadata.csv dataset_metadata.csv
mkdir -p column_metadata
cd column_metadata
files=(
    "2023%20Netflix%20Engagement%20Dataset_col_meta.csv"
    "Advertising%20Budget%20and%20Sales%20Dataset_col_meta.csv"
    "Bank%20Customer%20Churn%20Prediction%20Dataset_col_meta.csv"
    "Crop%20Production%20Dataset_col_meta.csv"
    "Customer%20Personality%20Analysis%20Dataset_col_meta.csv"
    "Dataset%20for%20Admission%20in%20the%20University_col_meta.csv"
    "Exercise%20and%20Fitness%20Metrics%20Dataset_col_meta.csv"
    "FIFA%20Official%20Dataset%202023_col_meta.csv"
    "Fruits%20and%20Vegetables%20Prices%20Dataset_col_meta.csv"
    "GPA%20Study%20Hours%20Dataset_col_meta.csv"
    "Grades%20of%20Students%20Dataset_col_meta.csv"
    "Green%20House%20Gas%20Produce%20by%20Different%20Industry_col_meta.csv"
    "Heart%20Failure%20Prediction%20Dataset_col_meta.csv"
    "Housing%20Price%20Dataset_col_meta.csv"
    "Income%20and%20Happiness%20Score%20Dataset_col_meta.csv"
    "Jobs%20and%20Salaries%20Dataset_col_meta.csv"
    "Los%20Angeles%20Library%20Monthly%20Statistics_col_meta.csv"
    "Lung%20Cancer%20Survey%20Dataset_col_meta.csv"
    "Monroe%20County%20Car%20Crach%20Dataset_col_meta.csv"
    "Netflix%20TV%20Shows%20and%20Movies%20Scores%20Dataset%20in%20IMDB_col_meta.csv"
    "Obesity%20or%20CVD%20Risk%20Dataset_col_meta.csv"
    "Pokedex%20Dataset_col_meta.csv"
    "Rainfall%20Dataset%20of%20Barak%20Velly_col_meta.csv"
    "Raman%20Ratings%20Dataset_col_meta.csv"
    "Real%20Estate%20Valuation%20Dataset_col_meta.csv"
    "Salary%20Dataset%20by%20Job%20Title%20and%20Country_col_meta.csv"
    "Stroke%20Prediction%20Dataset_col_meta.csv"
    "Student%20Information%20and%20Grades%20Dataset_col_meta.csv"
    "Student%20Performance%20Prediction%20Dataset_col_meta.csv"
    "Students%20Math-Reading-Writing%20Performance%20Dataset_col_meta.csv"
    "Students%20New%20MRW%20Score%20Dataset_col_meta.csv"
    "Terrorist%20Attacks%20Dataset_col_meta.csv"
    "US%20Christmas%20Tree%20Sales%20Dataset_col_meta.csv"
    "University%20Rank%20Dataset_col_meta.csv"
    "Used%20Car%20Dataset_col_meta.csv"
    "Video%20Games%20Sales%20Dataset_col_meta.csv"
    )
for file in "${files[@]}"; do wget -nc https://raw.githubusercontent.com/HKUSTDial/StatQA/refs/heads/main/Data/Metadata/Column%20Metadata/${file}; done
cd ..
mkdir -p processed_dataset
cd processed_dataset
files=(
    "2023%20Netflix%20Engagement%20Dataset.csv"
    "Advertising%20Budget%20and%20Sales%20Dataset.csv"
    "Bank%20Customer%20Churn%20Prediction%20Dataset.csv"
    "Crop%20Production%20Dataset.csv"
    "Customer%20Personality%20Analysis%20Dataset.csv"
    "Dataset%20for%20Admission%20in%20the%20University.csv"
    "Exercise%20and%20Fitness%20Metrics%20Dataset.csv"
    "FIFA%20Official%20Dataset%202023.csv"
    "Fruits%20and%20Vegetables%20Prices%20Dataset.csv"
    "GPA%20Study%20Hours%20Dataset.csv"
    "Grades%20of%20Students%20Dataset.csv"
    "Green%20House%20Gas%20Produce%20by%20Different%20Industry.csv"
    "Heart%20Failure%20Prediction%20Dataset.csv"
    "Housing%20Price%20Dataset.csv"
    "Income%20and%20Happiness%20Score%20Dataset.csv"
    "Jobs%20and%20Salaries%20Dataset.csv"
    "Los%20Angeles%20Library%20Monthly%20Statistics.csv"
    "Lung%20Cancer%20Survey%20Dataset.csv"
    "Monroe%20County%20Car%20Crach%20Dataset.csv"
    "Netflix%20TV%20Shows%20and%20Movies%20Scores%20Dataset%20in%20IMDB.csv"
    "Obesity%20or%20CVD%20Risk%20Dataset.csv"
    "Pokedex%20Dataset.csv"
    "Rainfall%20Dataset%20of%20Barak%20Velly.csv"
    "Raman%20Ratings%20Dataset.csv"
    "Real%20Estate%20Valuation%20Dataset.csv"
    "Salary%20Dataset%20by%20Job%20Title%20and%20Country.csv"
    "Stroke%20Prediction%20Dataset.csv"
    "Student%20Information%20and%20Grades%20Dataset.csv"
    "Student%20Performance%20Prediction%20Dataset.csv"
    "Students%20Math-Reading-Writing%20Performance%20Dataset.csv"
    "Students%20New%20MRW%20Score%20Dataset.csv"
    "Terrorist%20Attacks%20Dataset.csv"
    "US%20Christmas%20Tree%20Sales%20Dataset.csv"
    "University%20Rank%20Dataset.csv"
    "Used%20Car%20Dataset.csv"
    "Video%20Games%20Sales%20Dataset.csv"
    )
for file in "${files[@]}"; do wget -nc https://raw.githubusercontent.com/HKUSTDial/StatQA/refs/heads/main/Data/Processed%20Dataset/${file}; done
# Iterate through all CSV files in the current directory
for file in *.csv; do
    # Replace spaces with underscores in the filename
    new_name="${file// /_}"
    # Rename the file
    mv "$file" "$new_name"
done

echo "All CSV files have been renamed."
cd ..
cd ..


####################### Download QRData data
mkdir -p QRData
cd QRData
# Download data.zip
if [ ! -f "data.zip" ]; then
    wget https://github.com/xxxiaol/QRData/raw/refs/heads/main/benchmark/data.zip
else
    echo "data.zip already exists. Skipping download."
fi

# Unzip only if not already extracted
if [ ! -d "data" ]; then
    unzip data.zip
    rm data.zip
else
    echo "Data already extracted. Skipping unzip."
fi
wget -nc https://raw.githubusercontent.com/xxxiaol/QRData/refs/heads/main/benchmark/QRData.json
cd ..



####################### Download DiscoveryBench data
mkdir -p DiscoveryBench
cd DiscoveryBench
wget -nc https://raw.githubusercontent.com/allenai/discoverybench/refs/heads/main/eval/answer_key_real.csv
mkdir -p archaeology introduction_pathways_non-native_plants meta_regression
mkdir -p meta_regression_raw nls_incarceration nls_raw nls_ses
mkdir -p requirements_engineering_for_ML_enabled_systems worldbank_education_gdp worldbank_education_gdp_indicators
cd archaeology
files=("capital.csv"
    "metadata_0.json"
    "metadata_1.json"
    "metadata_10.json"
    "metadata_11.json"
    "metadata_12.json"
    "metadata_13.json"
    "metadata_14.json"
    "metadata_15.json"
    "metadata_16.json"
    "metadata_17.json"
    "metadata_18.json"
    "metadata_19.json"
    "metadata_2.json"
    "metadata_20.json"
    "metadata_21.json"
    "metadata_22.json"
    "metadata_23.json"
    "metadata_24.json"
    "metadata_25.json"
    "metadata_26.json"
    "metadata_27.json"
    "metadata_28.json"
    "metadata_29.json"
    "metadata_3.json"
    "metadata_30.json"
    "metadata_31.json"
    "metadata_32.json"
    "metadata_33.json"
    "metadata_34.json"
    "metadata_35.json"
    "metadata_36.json"
    "metadata_37.json"
    "metadata_4.json"
    "metadata_5.json"
    "metadata_6.json"
    "metadata_7.json"
    "metadata_8.json"
    "metadata_9.json"
    "pollen_openness_score_Belau_Woserin_Feeser_et_al_2019.csv"
    "time_series_data.csv"
    )
for file in "${files[@]}"; do wget -nc https://raw.githubusercontent.com/allenai/discoverybench/refs/heads/main/discoverybench/real/test/archaeology/${file}; done
cd ..
cd introduction_pathways_non-native_plants
files=("invaded_niche_pathways.csv"
       "invasion_success_pathways.csv"
       "metadata_0.json"
       "metadata_1.json"
       "metadata_2.json"
       "metadata_3.json"
       "metadata_4.json"
       "metadata_5.json"
       "phylogenetic_tree.txt"
       "temporal_trends_contingency_table.csv"
    )
for file in "${files[@]}"; do wget -nc https://raw.githubusercontent.com/allenai/discoverybench/refs/heads/main/discoverybench/real/test/introduction_pathways_non-native_plants/${file}; done
cd ..
cd meta_regression
files=("meta-regression_joined_data_heterogeneity_in_replication_projects.csv"
       "metadata_0.json"
       "metadata_1.json"
       "metadata_10.json"
       "metadata_11.json"
       "metadata_12.json"
       "metadata_13.json"
       "metadata_14.json"
       "metadata_15.json"
       "metadata_16.json"
       "metadata_17.json"
       "metadata_18.json"
       "metadata_19.json"
       "metadata_2.json"
       "metadata_3.json"
       "metadata_4.json"
       "metadata_5.json"
       "metadata_6.json"
       "metadata_7.json"
       "metadata_8.json"
       "metadata_9.json"
    )
for file in "${files[@]}"; do wget -nc https://raw.githubusercontent.com/allenai/discoverybench/refs/heads/main/discoverybench/real/test/meta_regression/${file}; done
cd ..
cd meta_regression_raw
files=("meta-regression_replication_success_data_heterogeneity_in_replication_projects.csv"
       "meta-regression_study_data_heterogeneity_in_replication_projects.csv"
       "metadata_0.json"
       "metadata_1.json"
       "metadata_10.json"
       "metadata_11.json"
       "metadata_12.json"
       "metadata_13.json"
       "metadata_14.json"
       "metadata_15.json"
       "metadata_16.json"
       "metadata_17.json"
       "metadata_18.json"
       "metadata_19.json"
       "metadata_2.json"
       "metadata_3.json"
       "metadata_4.json"
       "metadata_5.json"
       "metadata_6.json"
       "metadata_7.json"
       "metadata_8.json"
       "metadata_9.json"
    )
for file in "${files[@]}"; do wget -nc https://raw.githubusercontent.com/allenai/discoverybench/refs/heads/main/discoverybench/real/test/meta_regression_raw/${file}; done
cd ..
cd nls_incarceration
files=("metadata_0.json"
       "metadata_1.json"
       "metadata_10.json"
       "metadata_11.json"
       "metadata_12.json"
       "metadata_13.json"
       "metadata_14.json"
       "metadata_15.json"
       "metadata_2.json"
       "metadata_3.json"
       "metadata_4.json"
       "metadata_5.json"
       "metadata_6.json"
       "metadata_7.json"
       "metadata_8.json"
       "metadata_9.json"
       "nls_incarceration_processed.csv"
    )
for file in "${files[@]}"; do wget -nc https://raw.githubusercontent.com/allenai/discoverybench/refs/heads/main/discoverybench/real/test/nls_incarceration/${file}; done
cd ..
cd nls_raw
files=("metadata_2.json"
       "metadata_3.json"
       "metadata_4.json"
       "metadata_5.json"
       "metadata_6.json"
       "metadata_7.json"
       "metadata_8.json"
       "nls_raw.csv"
    )
for file in "${files[@]}"; do wget -nc https://raw.githubusercontent.com/allenai/discoverybench/refs/heads/main/discoverybench/real/test/nls_raw/${file}; done
cd ..
cd nls_ses
files=("metadata_0.json"
       "metadata_1.json"
       "metadata_10.json"
       "metadata_11.json"
       "metadata_2.json"
       "metadata_3.json"
       "metadata_4.json"
       "metadata_5.json"
       "metadata_6.json"
       "metadata_7.json"
       "metadata_8.json"
       "metadata_9.json"
       "nls_ses_processed.csv"
    )
for file in "${files[@]}"; do wget -nc https://raw.githubusercontent.com/allenai/discoverybench/refs/heads/main/discoverybench/real/test/nls_ses/${file}; done
cd ..
cd requirements_engineering_for_ML_enabled_systems
files=("metadata_0.json"
       "metadata_1.json"
       "metadata_10.json"
       "metadata_11.json"
       "metadata_12.json"
       "metadata_13.json"
       "metadata_14.json"
       "metadata_2.json"
       "metadata_3.json"
       "metadata_4.json"
       "metadata_5.json"
       "metadata_6.json"
       "metadata_7.json"
       "metadata_8.json"
       "metadata_9.json"
       "requirements_engineering_for_ML-enabled_systems.csv"
    )
for file in "${files[@]}"; do wget -nc https://raw.githubusercontent.com/allenai/discoverybench/refs/heads/main/discoverybench/real/test/requirements_engineering_for_ML_enabled_systems/${file}; done
cd ..
cd worldbank_education_gdp
files=("metadata_0.json"
       "metadata_1.json"
       "metadata_2.json"
       "metadata_3.json"
       "metadata_4.json"
       "worldbank_education_gdp.csv"
    )
for file in "${files[@]}"; do wget -nc https://raw.githubusercontent.com/allenai/discoverybench/refs/heads/main/discoverybench/real/test/worldbank_education_gdp/${file}; done
cd ..
cd worldbank_education_gdp_indicators
files=("Adjusted_savings_education_expenditure_percentage_of_GNI.csv"
       "Exports_of_goods_and_services_annual_percentage_growth.csv"
       "GNI_per_capita_constant_2015_USdollar.csv"
       "Labor_force_participation_rate_total_percentage_of_total_population_ages_15+_modeled_ILO_estimate.csv"
       "School_enrollment_primary_percentage_gross.csv"
       "School_enrollment_secondary_percentage_gross.csv"
       "metadata_0.json"
       "metadata_1.json"
       "metadata_2.json"
       "metadata_3.json"
       "metadata_4.json"
    )
for file in "${files[@]}"; do wget -nc https://raw.githubusercontent.com/allenai/discoverybench/refs/heads/main/discoverybench/real/test/worldbank_education_gdp_indicators/${file}; done
cd ..
