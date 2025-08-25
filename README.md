# Smart-Budget-Equalizer
a simple budget tracker app with ML model
Features Implemented:
📊 Budget Setup
Set income and select profile type (Student, Employee, Businessman)
Choose default or custom category allocations
Automatic proportional adjustment if category totals don’t match income

💸 Expense Tracking
Add expenses by category, amount, and description
ML-based overspending warnings per category

📑 Reports
Detailed breakdown of allocation, spent, and remaining per category
Savings updated automatically with leftover amounts

📊 Dashboard
Visual CLI bar charts for spending and savings
Predicted spending for next month using ML (Linear Regression)
Smart suggestions for adjusting categories if overspending is predicted

📤 Export & Visualization
Export expense history as CSV
Generate bar chart of spending per category (saved as PNG)

🤖 Machine Learning Integration
Decision Tree Classifier → Predicts overspending per category
Linear Regression → Forecasts next month’s category-wise spending
