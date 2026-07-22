import pandas as pd

csv_path = 'main_data.csv'

for encoding in ('utf-8', 'cp1252', 'latin-1'):
    try:
        data = pd.read_csv(csv_path, encoding=encoding)
        print(f'Successfully read {csv_path} with encoding: {encoding}')
        break
    except UnicodeDecodeError:
        continue
else:
    raise ValueError(f'Unable to read {csv_path} with a supported encoding')

# Test print to check the data shape
print(f'Data shape: {data.shape}')

# Calculate total percentage of OCR errors (Published != Captured)
total_errors = (data['Published'] != data['Captured']).mean() * 100
print(f'Total OCR Error Rate: {total_errors:.2f}%')
# Print first example of an OCR error with the row number
error_example = data[data['Published'] != data['Captured']].iloc[1]
print(f'Example OCR Error (Row {error_example.name+2}):\nPublished: {error_example["Published"]}\nCaptured: {error_example["Captured"]}\nConfidence: {error_example["confidence"]}')

# OCR error occurs and confidence above threshold
error = (data['Published'] != data['Captured']) & (data['confidence'] >= 0.8)
# Print first example of an OCR error with confidence above threshold
error_example_conf = data[error].iloc[1]
print(f'Example OCR Error with Confidence >= 0.8 (Row {error_example_conf.name+2}):\nPublished: {error_example_conf["Published"]}\nCaptured: {error_example_conf["Captured"]}\nConfidence: {error_example_conf["confidence"]}')
# Calculate error rate based on confidence threshold
# Error occurs when published != captured and confidence >= threshold
# for confidence_threshold in (0.75, 0.8, 0.85, 0.9, 0.95):
#     error_rate = (data['Published'] != data['Captured']) & (data['confidence'] >= confidence_threshold)
#     error_rate_percentage = error_rate.mean() * 100
#     print(f'Confidence Threshold: {confidence_threshold}, Error Rate: {error_rate_percentage:.2f}%')
