class A:
    def __init__(self, value):
        self.value = value

B = A

# Create two instances
instance1 = B(10)
instance2 = B(20)

# Check if they are different instances
print(instance1 is instance2)  # Output: False

# Check their values
print(instance1.value)  # Output: 10
print(instance2.value)  # Output: 20