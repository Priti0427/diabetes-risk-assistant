# Demo transcript

Generated 2026-11-15 by `python scripts/demo.py` in **offline rule-based** mode.

## Complete query

**User:** I'm a 35-year-old woman with a glucose level of 148 and a BMI of 33.6. What does your model predict?

**Assistant:** Based on the information you provided, the model estimates a **75% probability** of diabetes, which it classifies as **elevated risk** (prediction: positive).

Key factors: your plasma glucose (mg/dL) of 148 raises the estimate by about 15 points versus a typical value of 117; your age (years) of 35 raises the estimate by about 1 points versus a typical value of 29.

Your BMI falls in the 'obese' category.

I did not have your number of pregnancies, diastolic blood pressure (mm Hg), triceps skin-fold thickness (mm), 2-hour serum insulin (mu U/mL), diabetes pedigree function, so typical values were used; providing them would make the estimate more personal.

This is a statistical screening estimate from a model trained on a small historical dataset of Pima women, not a medical diagnosis. Please confirm with a clinician, for example with an HbA1c or fasting glucose test.

<details><summary>Parsed input (parser: rules, intent: risk_assessment)</summary>

- features: `{'Glucose': 148.0, 'BMI': 33.6, 'Age': 35.0}`
- model probability: **0.747** (elevated risk), label=1
- filled with typical values: `['Pregnancies', 'BloodPressure', 'SkinThickness', 'Insulin', 'DiabetesPedigreeFunction']`
- top contributions: Glucose +0.152, Age +0.014, BMI +0.002

</details>

## Rich query with units to convert

**User:** I'm 52, blood pressure 130/85, glucose 110 mg/dL, 5'4" and 150 lbs, two pregnancies, insulin 94.

**Assistant:** Some values don't look right (number of pregnancies: 94  is outside the plausible range 0-20). Could you double-check them?

<details><summary>Parsed input (parser: rules, intent: risk_assessment)</summary>

- features: `{'Glucose': 110.0, 'BloodPressure': 85.0, 'Insulin': 94.0, 'BMI': 25.7, 'Age': 52.0}`
- invalid: `{'Pregnancies': '94  is outside the plausible range 0-20'}`

</details>

## Low-risk profile

**User:** 23 year old, never been pregnant, glucose 85, BMI 21, blood pressure 110/65.

**Assistant:** Some values don't look right (number of pregnancies: 85  is outside the plausible range 0-20). Could you double-check them?

<details><summary>Parsed input (parser: rules, intent: risk_assessment)</summary>

- features: `{'Glucose': 85.0, 'BloodPressure': 65.0, 'BMI': 21.0, 'Age': 23.0}`
- invalid: `{'Pregnancies': '85  is outside the plausible range 0-20'}`

</details>

## Edge case: incomplete input

**User:** I'm 45 and my BMI is 28. What's my diabetes risk?

**Assistant:** I still need your plasma glucose (mg/dL) before I can run the model. So far I have: body mass index (kg/m²) = 28, age (years) = 45.

<details><summary>Parsed input (parser: rules, intent: risk_assessment)</summary>

- features: `{'BMI': 28.0, 'Age': 45.0}`
- missing required: `['Glucose']`

</details>

## Edge case: implausible value

**User:** glucose 5000, bmi 30, age 40

**Assistant:** Some values don't look right (plasma glucose (mg/dL): 5000 mg/dL is outside the plausible range 40-400). Could you double-check them?

<details><summary>Parsed input (parser: rules, intent: risk_assessment)</summary>

- features: `{'BMI': 30.0, 'Age': 40.0}`
- invalid: `{'Glucose': '5000 mg/dL is outside the plausible range 40-400'}`

</details>

## Edge case: out of scope

**User:** What's the weather like in Boston today?

**Assistant:** I can only help with one thing: estimating type 2 diabetes risk from health measurements using a model trained on the Pima Indians Diabetes dataset. That question is outside what I can answer. If you'd like a risk estimate, tell me your plasma glucose (mg/dL), BMI (or height and weight), and age.

<details><summary>Parsed input (parser: rules, intent: out_of_scope)</summary>

- features: `{}`

</details>

## Edge case: asks about the tool

**User:** What information do you need from me?

**Assistant:** I estimate the probability of type 2 diabetes using a machine-learning model. To run it I need at least your **glucose level (mg/dL)**, **BMI** (or height and weight) and **age**. Optionally you can add diastolic blood pressure, 2-hour insulin, triceps skin-fold thickness, number of pregnancies and the diabetes pedigree function (family-history score). Anything you leave out is filled with a typical value, which makes the estimate less personal. For example: "I'm a 45-year-old woman, glucose 130, BMI 29, blood pressure 120/82."

<details><summary>Parsed input (parser: rules, intent: about_tool)</summary>

- features: `{}`
- missing required: `['Glucose', 'BMI', 'Age']`

</details>
