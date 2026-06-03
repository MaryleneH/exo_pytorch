import torch
import torch.nn as nn
import torch.optim as optim


distances = torch.tensor([[1.0], [2.0], [3.0], [4.0]], dtype=torch.float32)
times = torch.tensor([[6.96], [12.11], [16.77], [22.21]], dtype=torch.float32)

model = nn.Sequential(nn.Linear(1,1))

loss_function = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Training loop

for epoch in range(500):
    # 0. Reset the optimizer
    optimizer.zero_grad()

    # 1. Make predictions
    outputs = model(distances)

    # 2. Calculate the loss 
    loss = loss_function(outputs, times)

    # 3. Calculate adjustments 
    loss.backward()

    # 4. Update the model
    optimizer.step()

    # Print loss every 50 epochs
    if (epoch + 1) % 50 == 0:
        print(f"Epoch {epoch + 1}: Loss = {loss.item()}")

## Make a prediction

with torch.no_grad():
    test_distance = torch.tensor([[7.0]], dtype=torch.float32)
    predicted_time = model(test_distance)
    print(f"Predicted time for 25 miles : {predicted_time.item():.1f} minutes")


# Inspecting the model

# Access the first (and only) layer in the sequential model
layer = model[0]

# Get weights and bias
weights = layer.weight.data.numpy()
bias = layer.bias.data.numpy()

print(f"Weight: {weights}")
print(f"Bias: {bias}")