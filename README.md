<div align='left'>
  

  <h1>Health Check</h1>

  <p>
The Diabetes Prediction App is a tool that predicts the probability of a patient having diabetes based on diagnostic measurements. This tool is intended for Patients above the age of 21 years, of Pima Indian heritage, and uses a dataset from the Pima Indian Dataset.
  </p>
  

<!-- Badges -->

<!-- Table of Contents -->

## :notebook_with_decorative_cover: Table of Contents

- [Dataset](#signal_strength-dataset)
- [Dependencies](#toolbox-dependecies)
- [Installation](#gear-installation)
- [Usage](#play_or_pause_button-usage)
- [Inputs](#construction-inputs)
- [Outputs](#rocket-outputs)
- [Deployment and Notebook](#triangular_flag_on_post-deployment-and-notebook)
- [Contact](#handshake-contact)



## :signal_strength: Dataset

The trained dataset is originally from the Pima Indian Dataset. The objective is to predict based on diagnostic measurements whether a patient has diabetes. Several constraints were placed on the selection of these instances from a larger database. It includes following health criteria:

- Glucose: Plasma glucose concentration a 2 hours in an oral glucose tolerance test
- BloodPressure: Diastolic blood pressure (mm Hg)
- SkinThickness: Triceps skin fold thickness (mm)
- Insulin: 2-Hour serum insulin (mu U/ml)
- BMI: Body mass index (weight in kg/(height in m)^2)
- DiabetesPedigreeFunction: Diabetes pedigree function
- Age: Age (years)
- Outcome: Class variable (0 or 1)

### Details
- Number of Instances: 768
- Number of Attributes: 8 plus class
- Missing Attribute Values: Yes
  

## :toolbox: Dependecies

`streamlit==1.35.0`

`pandas==1.3.3`

`numpy==1.21.2`

`matplotlib==3.4.3`

`plotly==5.3.1`

`seaborn==0.11.2`

`scikit-learn==0.24.2`


## :gear: Installation

Clone the repository and install the required dependencies using the following commands:

```bash
git clone https://github.com/codewithakas/Health-Check.git
```

```bash
cd client/app.py

```

```bash
pip install -r requirements.txt
```

```bash
streamlit run app.py
```

## :play_or_pause_button: Usage

1. Open the app in your web browser.
2. Enter the required information in the input fields.
3. Click the 'Predict' button to generate the prediction.



## :construction: Inputs
Click on the link and reboot the tool or run locally and enter your:

* Glucose: Plasma glucose concentration a 2 hours in an oral glucose tolerance test
* Blood Pressure: Diastolic blood pressure (mm Hg)
* Skin Thickness: Triceps skin fold thickness (mm)
* Insulin: 2-Hour serum insulin (mu U/ml)
* BMI: Body mass index (weight in kg/(height in m)^2)
* Diabetes Pedigree Function: Diabetes pedigree function
* Age: Age (years)



## :rocket: Outputs
The app will display one of the following messages:

* "The Patient is not likely to have Diabetes.




## :triangular_flag_on_post: Deployment and Notebook

This tool has been deployed using [`Streamlit`](https://streamlit.io/). Learn about streamlit deployment [`here`](https://docs.streamlit.io/streamlit-community-cloud/get-started/deploy-an-app). 




## :handshake: Contact

Project Link: [https://github.com/codewithakas/Health-Check](https://github.com/codewithakas/Health-Check)
<hr />
<br />
<div align="center">Don't forget to leave a star ⭐️</div>
