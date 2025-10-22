import numpy as np

# Load the data
data = np.loadtxt("spamTrain1.csv", delimiter=",")

# Shuffle the data randomly
np.random.seed(0)
indices = np.random.permutation(len(data))
shuffled_data = data[indices]

# Calculate split sizes
total_rows = len(shuffled_data)
final_test_size = int(0.20 * total_rows)
start_test_size = int(0.40 * total_rows)
# Remaining goes to data

# Split the data
final_test = shuffled_data[:final_test_size]
start_test = shuffled_data[final_test_size:final_test_size + start_test_size]
train_data = shuffled_data[final_test_size + start_test_size:]

# Store the original indices for printing
final_test_indices = indices[:final_test_size]
start_test_indices = indices[final_test_size:final_test_size + start_test_size]
train_data_indices = indices[final_test_size + start_test_size:]

# Save to files
np.savetxt("spamTrainFinalTest.csv", final_test, delimiter=",")
np.savetxt("splitTrainStartTest.csv", start_test, delimiter=",")
np.savetxt("splitTrainData.csv", train_data, delimiter=",")

# Print row indices (original row numbers from spamTrain1.csv)
print("Final Test rows:")
print(", ".join(str(i) for i in sorted(final_test_indices)))

print("\nStart Test rows:")
print(", ".join(str(i) for i in sorted(start_test_indices)))

print("\nData:")
print(", ".join(str(i) for i in sorted(train_data_indices)))

print(f"\n--- Summary ---")
print(f"Total rows: {total_rows}")
print(f"Final Test: {len(final_test)} rows (20%)")
print(f"Start Test: {len(start_test)} rows (40%)")
print(f"Train Data: {len(train_data)} rows (40%)")
