import pandas as pd
import os

# Create dataset folder if it doesn't exist
os.makedirs("data", exist_ok=True)

# Sample synthetic requirements
data = {
    "statement": [
        "The system shall allow users to log in using email.",
        "The interface must be user-friendly.",
        "System should respond within 2 seconds.",
        "Data must be encrypted at rest.",
        "Users should be able to reset their passwords.",
        "System must comply with GDPR regulations.",
        "The system shall send email notifications on password reset.",
        "The login page shall have a 2-factor authentication option.",
        "System should be available 99.9% of the time.",
        "All transactions shall be logged for audit purposes.",
        "The UI must be accessible to users with visual impairments.",
        "System should scale to support 10,000 concurrent users.",
        "Users should be able to change their username and password.",
        "The system shall maintain session data for 30 minutes.",
        "The API shall support JSON and XML formats.",
    ],
    "label": [
        0, 1, 1, 1, 0, 1,
        0, 0, 1, 0, 1, 1,
        0, 0, 0
    ]  # 0 = Functional, 1 = Non-Functional
}

df = pd.DataFrame(data)
df.to_csv("data/requirements.csv", index=False)

print("Dataset saved to data/requirements.csv")
