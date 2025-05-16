from folktables import ACSDataSource, ACSEmployment

for state in ["CA", "TX", "FL" "IL"]:
    data_source = ACSDataSource(survey_year='2018', horizon='1-Year', survey='person')
    acs_data = data_source.get_data(states=[state], download=True)
    acs_data.to_csv(f'../empirical/data/{state}_2018_1yr.csv', index=False)