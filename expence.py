import json
import os
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder

CONFIG_FILE = "budget_config.json"
EXPENSE_FILE = "expenses.json"

# ---------------- Helpers ---------------- #
def load_config():
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, "r") as f:
            return json.load(f)
    return {}

def save_config(config):
    with open(CONFIG_FILE, "w") as f:
        json.dump(config, f, indent=4)

def load_expenses():
    if os.path.exists(EXPENSE_FILE):
        with open(EXPENSE_FILE, "r") as f:
            return json.load(f)
    return []

def save_expenses(expenses):
    with open(EXPENSE_FILE, "w") as f:
        json.dump(expenses, f, indent=4)

# ---------------- ML: Overspending Warning ---------------- #
def train_overspend_model():
    expenses = load_expenses()
    config = load_config()
    if not expenses or not config:
        return None, None
    data = []
    for e in expenses:
        month = e.get("month_idx", datetime.now().month)
        cat = e["category"]
        amt = e["amount"]
        allocated = config["allocations"].get(cat, 0)
        overspend = 1 if amt > allocated else 0
        data.append({"category": cat, "amount": amt, "month": month, "profile": config["profile"], "overspend": overspend})
    df = pd.DataFrame(data)
    le_cat = LabelEncoder()
    le_prof = LabelEncoder()
    df["category_enc"] = le_cat.fit_transform(df["category"])
    df["profile_enc"] = le_prof.fit_transform(df["profile"])
    X = df[["category_enc", "amount", "profile_enc"]]
    y = df["overspend"]
    clf = DecisionTreeClassifier()
    clf.fit(X, y)
    return clf, (le_cat, le_prof)

def check_overspend(category, amount):
    clf, encoders = train_overspend_model()
    if clf is None:
        return False
    le_cat, le_prof = encoders
    config = load_config()
    cat_enc = le_cat.transform([category])[0] if category in le_cat.classes_ else 0
    prof_enc = le_prof.transform([config["profile"]])[0]
    pred = clf.predict([[cat_enc, amount, prof_enc]])[0]
    return pred == 1

# ---------------- ML: Next Month Prediction ---------------- #
def predict_next_month():
    expenses = load_expenses()
    config = load_config()
    if not expenses or not config:
        print("No data to predict yet.")
        return {}

    data = []
    for e in expenses:
        month_idx = int(e.get("month_idx", datetime.now().month))
        data.append({"category": e["category"], "month_idx": month_idx, "amount": e["amount"], "profile": config["profile"]})

    df = pd.DataFrame(data)
    le_cat = LabelEncoder()
    df["category_enc"] = le_cat.fit_transform(df["category"])
    le_prof = LabelEncoder()
    df["profile_enc"] = le_prof.fit_transform(df["profile"])

    predictions = {}
    for cat in df["category"].unique():
        cat_df = df[df["category"] == cat]
        X = cat_df[["month_idx", "profile_enc"]]
        y = cat_df["amount"]
        if len(X) < 2:
            predictions[cat] = y.iloc[-1]
            continue
        model = LinearRegression()
        model.fit(X, y)
        next_month = (datetime.now().month % 12) + 1
        pred = model.predict([[next_month, le_prof.transform([config["profile"]])[0]]])[0]
        predictions[cat] = max(pred, 0)
    print("\n Predicted Spending for Next Month:")
    for cat, amt in predictions.items():
        print(f"- {cat}: {round(amt,2)}")
    return predictions

# ---------------- Automatic Budget Adjustment ---------------- #
def adjust_budget_suggestions():
    config = load_config()
    predictions = predict_next_month()
    suggestions = {}
    for cat, predicted in predictions.items():
        allocated = config["allocations"].get(cat, 0)
        if predicted > allocated:
            suggestions[cat] = round(predicted - allocated, 2)
    if suggestions:
        print("\n Suggested Adjustments for Next Month:")
        for cat, extra in suggestions.items():
            print(f"- Increase budget for {cat} by {extra}")
    else:
        print("\n Your current budget is sufficient for predicted spending.")

# ---------------- Setup Budget ---------------- #
def setup_budget():
    income = float(input(" Enter your monthly income: "))
    print("\nChoose your profile type:\n1. Student\n2. Employee\n3. Businessman")
    profile = input("Enter choice (1/2/3): ")
    if profile == "1":
        allocations = {"food": 0.40, "transport": 0.20, "shopping": 0.10, "other": 0.10}
    elif profile == "2":
        allocations = {"food": 0.30, "transport": 0.15, "shopping": 0.20, "other": 0.15}
    else:
        allocations = {"food": 0.20, "transport": 0.10, "shopping": 0.15, "other": 0.25}
    customize = input("\nCustomize categories? (y/n): ")
    if customize.lower() == "y":
        allocations = {}
        total = 0
        while True:
            cat = input("Enter category (or 'done'): ")
            if cat.lower() == "done":
                break
            amt = float(input(f"Amount for {cat}: "))
            allocations[cat] = amt
            total += amt
        if total != income:
            factor = income / total
            for cat in allocations:
                allocations[cat] *= factor
    else:
        allocations = {cat: round(income * frac,2) for cat, frac in allocations.items()}
    config = {"income": income, "allocations": allocations, "savings":0, "profile": profile}
    save_config(config)
    print("\n Budget setup complete!")
    adjust_budget_suggestions()

# ---------------- Add Expense ---------------- #
def add_expense():
    config = load_config()
    if not config:
        print(" Run setup first.")
        return
    category = input(f"Enter category {list(config['allocations'].keys())}: ")
    amount = float(input("Enter expense amount: "))
    description = input("Enter description (optional): ")

    if check_overspend(category, amount):
        print(f" Warning: You are likely to overspend in {category}!")

    now = datetime.now()
    month_idx = now.month
    expenses = load_expenses()
    expenses.append({"amount": amount, "category": category, "description": description, "month_idx": month_idx})
    save_expenses(expenses)
    print(f" Added {amount} to {category} ({description})")
    adjust_budget_suggestions()

# ---------------- Show Report ---------------- #
def show_report():
    config = load_config()
    if not config:
        print(" Run setup first.")
        return
    expenses = load_expenses()
    spent_per_category = {}
    for e in expenses:
        spent_per_category[e["category"]] = spent_per_category.get(e["category"], 0) + e["amount"]
    leftovers = 0
    print("\n--- Budget Report ---")
    for cat, allocated in config["allocations"].items():
        spent = spent_per_category.get(cat, 0)
        remaining = allocated - spent
        if remaining > 0:
            leftovers += remaining
        print(f"{cat}: allocated {allocated}, spent {spent}, remaining {remaining}")
    config["savings"] += leftovers
    save_config(config)
    print("\n Total Savings:", config["savings"])
    adjust_budget_suggestions()

# ---------------- Dashboard ---------------- #
def show_dashboard():
    config = load_config()
    if not config:
        print(" Run setup first.")
        return

    expenses = load_expenses()
    spent_per_category = {}
    for e in expenses:
        spent_per_category[e["category"]] = spent_per_category.get(e["category"], 0) + e["amount"]

    print("\n Budget Dashboard")
    print("-" * 40)
    for cat, allocated in config["allocations"].items():
        spent = spent_per_category.get(cat, 0)
        remaining = allocated - spent
        spent_bar = int((spent / allocated) * 20) if allocated > 0 else 0
        remaining_bar = 20 - spent_bar
        print(f"{cat}: [{'█' * spent_bar}{'.' * remaining_bar}] {spent}/{allocated} spent")
    print("-" * 40)

    # Savings Bar
    savings = config.get("savings", 0)
    total_income = config.get("income", 1)
    savings_bar = int((savings / total_income) * 20)
    print(f"Savings: [{'█' * savings_bar}{'.' * (20 - savings_bar)}] {savings}/{total_income}")
    
    # Show predicted spending
    predictions = predict_next_month()
    print("\n Predicted Spending Next Month")
    for cat, predicted in predictions.items():
        allocated = config["allocations"].get(cat, 0)
        bar_length = 20
        pred_bar = int((predicted / allocated) * bar_length) if allocated > 0 else 0
        if pred_bar > bar_length:
            pred_bar = bar_length
        print(f"{cat}: [{'█' * pred_bar}{'.' * (bar_length - pred_bar)}] Predicted: {round(predicted,2)}, Allocated: {allocated}")
    print("-" * 40)

# ---------------- Export Reports ---------------- #
def export_csv_report():
    expenses = load_expenses()
    if not expenses:
        print(" No expenses to export.")
        return
    df = pd.DataFrame(expenses)
    filename = f"expense_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    df.to_csv(filename, index=False)
    print(f" CSV report exported: {filename}")

def export_spending_chart():
    expenses = load_expenses()
    config = load_config()
    if not expenses or not config:
        print(" Not enough data for chart.")
        return

    spent_per_category = {}
    for e in expenses:
        spent_per_category[e["category"]] = spent_per_category.get(e["category"], 0) + e["amount"]

    categories = list(spent_per_category.keys())
    amounts = [spent_per_category[cat] for cat in categories]

    plt.figure(figsize=(8,5))
    plt.bar(categories, amounts, color='skyblue')
    plt.title("Spending per Category")
    plt.xlabel("Category")
    plt.ylabel("Amount Spent")
    plt.savefig("spending_chart.png")
    plt.show()
    print(" Spending chart saved as spending_chart.png")


# ---------------- Main Menu ---------------- #
def main():
    while True:
        print("\n--- Budget Manager ---")
        print("1. Setup Budget")
        print("2. Add Expense")
        print("3. Show Report")
        print("4. Show Dashboard")
        print("5. Export Reports")
        print("6. Export Spending Chart")
        print("7. Exit")
        choice = input("Choose an option: ")
        if choice == "1":
            setup_budget()
            show_dashboard()
        elif choice == "2":
            add_expense()
            show_dashboard()
        elif choice == "3":
            show_report()
        elif choice == "4":
            show_dashboard()
        elif choice == "5":
            export_csv_report()
        elif choice == "6":
            export_spending_chart()
        elif choice == "7":
            print(" Goodbye!")
            break
        else:
            print(" Invalid choice. Try again.")

if __name__ == "__main__":
    main()
