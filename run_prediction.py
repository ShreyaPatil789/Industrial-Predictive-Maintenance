from predictor import predict_machine_failure

sample_machine = {
    "Air temperature [K]": 300,
    "Process temperature [K]": 310,
    "Rotational speed [rpm]": 1500,
    "Torque [Nm]": 45,
    "Tool wear [min]": 200,
    "Type_L": 0,
    "Type_M": 1
}

result = predict_machine_failure(sample_machine)

print("\nPrediction Result:")
print(result)