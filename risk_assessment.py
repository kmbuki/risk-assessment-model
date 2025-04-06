import streamlit as st
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines

# QUESTIONS FOR LIKELIHOOD OF FAILURE
Q1 = "Is the project scope clearly defined and agreed upon by all stakeholders?"
Q2 = "Are the project goals realistic and achievable within the given time and resources?"
Q3 = "Does the project team have the necessary skills and capacity to deliver the project?"
Q4 = "Is there strong and consistent support from key stakeholders or leadership?"
Q5 = "Is the project adequately funded, and is the budget stable and secure?"
Q6 = "Are the required tools, technology, and infrastructure in place and accessible?"
Q7 = "Are there critical dependencies or external factors that could delay the project?"
Q8 = "Is there a contingency plan in place for potential delays or disruptions?"
Q9 = "Has a comprehensive risk assessment been conducted and documented?"
Q10 = "Is there a monitoring and evaluation framework to track progress and detect issues early?"

# QUESTIONS FOR CONSEQUENCE OF FAILURE
Q11 = "What would be the impact on beneficiaries, clients, or end-users if the project fails?"
Q12 = "Would project failure result in significant financial loss or wasted resources?"
Q13 = "Could failure damage the reputation or credibility of the organization or partners?"
Q14 = "Would project failure lead to legal or regulatory consequences?"
Q15 = "Could failure affect future funding opportunities or strategic partnerships?"

# ANSWERS FOR LIKELIHOOD OF FAILURE
A1 = ["Yes – very clear", "Somewhat clear", "Unclear", "Not defined at all"]
A2 = ["Completely achievable", "Somewhat achievable", "Barely achievable", "Not achievable"]
A3 = ["Fully skilled and staffed", "Some gaps", "Significant gaps", "Lacks core skills or capacity"]
A4 = ["Strong and consistent support", "Occasional support", "Limited support", "No support"]
A5 = ["Fully funded and stable", "Mostly funded", "Unstable or partial funding", "Underfunded"]
A6 = ["All in place", "Partially in place", "Some gaps", "Largely missing"]
A7 = ["No major dependencies", "Some dependencies", "High dependency on few elements", "Critical and uncertain dependencies"]
A8 = ["Comprehensive contingency plan", "Basic plan", "In development", "No contingency plan"]
A9 = ["Yes – up-to-date", "Yes – outdated", "Partial risk assessment", "No risk assessment"]
A10 = ["Robust and active", "Basic M&E setup", "Limited M&E", "No M&E framework"]

# ANSWERS FOR CONSEQUENCE OF FAILURE
A11 = ["N/A", "Negligible", "Minor inconvenience", "Some disruption", "Major disruption", "Critical harm"]
A12 = ["N/A", "None", "Minor loss", "Moderate loss", "Major financial loss", "Severe long-term loss"]
A13 = ["N/A", "No impact", "Minimal impact", "Some reputational risk", "Significant damage", "Severe damage"]
A14 = ["N/A", "None", "Low risk", "Moderate risk", "High risk", "Certain legal action"]
A15 = ["N/A", "No effect", "Minimal effect", "Some negative effect", "Likely loss of funding/partners", "Severe long-term impact"]

# QUESTION AND ANSWER GROUPINGS
qlist_likelihood = [Q1, Q2, Q3, Q4, Q5, Q6, Q7, Q8, Q9, Q10]
alist_likelihood = [A1, A2, A3, A4, A5, A6, A7, A8, A9, A10]
qlist_consequence = [Q10, Q11, Q12, Q13, Q14, Q15]
alist_consequence = [A10, A11, A12, A13, A14, A15]
    
with st.sidebar:
    st.header('Risk Factors Affecting the Likelihood of Failure')
    likelihood_total = 0
    likelihood_answers = 0
    for idx, question in enumerate(qlist_likelihood):
        ask = qlist_likelihood[idx]
        answer = alist_likelihood[idx]
        box = st.radio(ask, answer, index=3)
        user_answer = answer.index(box)
        likelihood_total = user_answer + likelihood_total
        if box == "N/A":
            likelihood_answers = 0 + likelihood_answers
        else:
            likelihood_answers = 1 + likelihood_answers
    
    st.divider()
    
    st.header('Risk Factors Affecting the Consequence of Failure')
    consequence_total = 0
    consequence_answers = 0
    for idx, question in enumerate(qlist_consequence):
        ask = qlist_consequence[idx]
        answer = alist_consequence[idx]
        box = st.select_slider(ask, answer, value=answer[3])
        user_answer = answer.index(box)
        consequence_total = user_answer + consequence_total
        if box == "N/A":
            consequence_answers = 0 + consequence_answers
        else:
            consequence_answers = 1 + consequence_answers
    st.divider()
    st.markdown('Version 0.0.3')
    st.markdown('Developed by D.M. Budd, P.Eng.')

if likelihood_answers == 0:
    likelihood_answers = 1

if consequence_answers == 0:
    consequence_answers = 1

# User risk score is reported based on the actual calculated number
# Axis have been switched as per EGBC Version 4.0
x_pt = consequence_total/consequence_answers
y_pt = likelihood_total/likelihood_answers

x_user = [x_pt]
y_user = [y_pt]

# Determine the risk assessment category that is reported
if x_pt >= 0:
    if y_pt <= 3:
        report_ra = "Low Risk"
    else:
        report_ra = "Moderate Risk"
if x_pt >= 1:
    if y_pt <= 2:
        report_ra = "Low Risk"
    elif y_pt <= 4:
        report_ra = "Moderate Risk"
    else:
        report_ra = "High Risk"
if x_pt >= 2:
    if y_pt <= 1:
        report_ra = "Low Risk"
    elif y_pt <= 3:
        report_ra = "Moderate Risk"
    else:
        report_ra = "High Risk"
if x_pt >= 3:
    if y_pt <= 2:
        report_ra = "Moderate Risk"
    elif y_pt <= 4:
        report_ra = "High Risk"
    else:
        report_ra = "Very High Risk"
if x_pt >= 4:
    if y_pt <= 1:
        report_ra = "Moderate Risk"
    elif y_pt <= 3:
        report_ra = "High Risk"
    else:
        report_ra = "Very High Risk"

# Review the risk assessment category if the value is on a border category
# This code can be cleaned up in the future
if x_pt <= 1 and y_pt == 3:
    report_ra = "Low-to-Moderate Risk"
elif x_pt <= 2 and x_pt >= 1 and y_pt == 2:
    report_ra = "Low-to-Moderate Risk"
elif x_pt <= 3 and x_pt >= 2 and y_pt == 1:
    report_ra = "Low-to-Moderate Risk"

elif x_pt == 1 and y_pt >= 2 and y_pt <= 3:
    report_ra = "Low-to-Moderate Risk"
elif x_pt == 2 and y_pt >= 1 and y_pt <= 2:
    report_ra = "Low-to-Moderate Risk"
elif x_pt == 3 and y_pt >= 0 and y_pt <= 1:
    report_ra = "Low-to-Moderate Risk"

elif x_pt <= 2 and x_pt >= 1 and y_pt == 4:
    report_ra = "Moderate-to-High Risk"
elif x_pt <= 3 and x_pt >= 2 and y_pt == 3:
    report_ra = "Moderate-to-High Risk"
elif x_pt <= 4 and x_pt >= 3 and y_pt == 2:
    report_ra = "Moderate-to-High Risk"
elif x_pt <= 5 and x_pt >= 4 and y_pt == 1:
    report_ra = "Moderate-to-High Risk"

elif x_pt == 1 and y_pt >= 4 and y_pt <= 5:
    report_ra = "Moderate-to-High Risk"
elif x_pt == 2 and y_pt >= 3 and y_pt <= 4:
    report_ra = "Moderate-to-High Risk"
elif x_pt == 3 and y_pt >= 2 and y_pt <= 3:
    report_ra = "Moderate-to-High Risk"
elif x_pt == 4 and y_pt >= 1 and y_pt <= 2:
    report_ra = "Moderate-to-High Risk"

elif x_pt <= 4 and x_pt >= 3 and y_pt == 4:
    report_ra = "High-to-Very High Risk"
elif x_pt <= 5 and x_pt >= 4 and y_pt == 3:
    report_ra = "High-to-Very High Risk"

elif x_pt == 3 and y_pt >= 4 and y_pt <= 5:
    report_ra = "High-to-Very High Risk"
elif x_pt == 4 and y_pt >= 3 and y_pt <= 4:
    report_ra = "High-to-Very High Risk"
else:
    report_ra = report_ra


# Normalized score is based on rounding to the nearest whole number
x_pt_norm = round(x_pt,0)
y_pt_norm = round(y_pt,0)

# Not used in plot
# x_user_norm = [x_pt_norm]
# y_user_norm = [y_pt_norm]


# Determine the Consequence of Failure category
if x_pt_norm == 5:
    report_consequence = "Very High"
elif x_pt_norm == 4:
    report_consequence = "High"
elif x_pt_norm == 3:
    report_consequence = "Medium"
elif x_pt_norm == 2:
    report_consequence = "Low"
else:
    report_consequence = "Very Low"

    
# Determine the Likelihood of Failure category
if y_pt_norm == 5:
    report_likelihood = "Highly Likely"
elif y_pt_norm == 4:
    report_likelihood = "Likely"
elif y_pt_norm == 3:
    report_likelihood = "Possible"
elif y_pt_norm == 2:
    report_likelihood = "Unlikely"
else:
    report_likelihood = "Rare"


# SET DEFAULT APPEARANCE OF MATRIX PLOT
fig, ax = plt.subplots()
ax.set_xlabel('Consequence of Failure')
ax.set_ylabel('Likelihood of Failure')
plt.xlim([1,5])
plt.ylim([1,5])
plt.xticks([0,1,2,3,4,5])
plt.yticks([0,1,2,3,4,5])
plt.grid(linestyle='--', linewidth=0.5)

# SET BASE VALUES AS PER TABLE B-1 RISK ASSESSMENT MATRIX
x = np.array([0,1,1,2,2,3,3,4,4,5])
y_low = np.array([3,3,2,2,1,1,0,0,0,0])
y_mod = np.array([5,5,4,4,3,3,2,2,1,1])
y_high = np.array([5,5,5,5,5,5,4,4,3,3])
y_vhigh = np.array([5,5,5,5,5,5,5,5,5,5])

if x_pt >= 3:
    x_text_offset = -45
else:
    x_text_offset = 45

if y_pt >= 3:
    y_text_offset = -45
else:
    y_text_offset = 45


# PLOTTING
ax.set_title('Table B-1 Risk Assessment Matrix (EGBC)')
ax.plot(x_user, y_user, color='black', marker='o', markersize=8)
# Removed normalized score from being plotted
# ax.plot(x_user_norm, y_user_norm, marker='s', color='blue', markersize=8, markerfacecolor='none', markeredgecolor='blue')
ax.annotate(
    'Risk Score', 
    xy=(x_pt,y_pt), xycoords='data', 
    xytext=(x_pt + x_text_offset, y_pt + y_text_offset), textcoords='offset points', 
    arrowprops=dict(arrowstyle='->',
                    connectionstyle='arc3, rad=.2'))
ax.stackplot(x, y_vhigh, color='red')
ax.stackplot(x, y_high, color='orange')
ax.stackplot(x, y_mod, color='yellow')
ax.stackplot(x, y_low, color='green')

red_patch = mpatches.Patch(color='red', label='Very High')
orange_patch = mpatches.Patch(color='orange', label='High')
yellow_patch = mpatches.Patch(color='yellow', label='Moderate')
green_patch = mpatches.Patch(color='green', label='Low')
black_circle = mlines.Line2D([], [], marker='o', color='black', markersize=8, linestyle='None', label='Risk Score')
# Removed normalized score from being plotted
# blue_square = mlines.Line2D([], [], marker='s', color='blue', markersize=8, 
#                            markerfacecolor='none', markeredgecolor='blue', 
#                            linestyle='None', label='Normalized Score')
plt.legend(handles=[red_patch, orange_patch, yellow_patch, green_patch, black_circle], bbox_to_anchor=(1.05,1), loc=2, borderaxespad=0.)
my_fig = plt.show()


# TEXT OUTPUT
# Reporting for Consequence and Likelihood of Failure represent the whole number values
# Reporting for the Risk Assessment Category is based on the decimal values
# If the Category is on the border, it is assessed as Category1-to-Category2
st.pyplot(fig)
st.markdown(f"Consequence of Failure = **{round(x_pt,1)}** or **{report_consequence}**")
st.markdown(f"Likelihood of Failure = **{round(y_pt,1)}** or **{report_likelihood}**")
st.markdown(f"The Risk Assessment score is **({int(x_pt_norm)}, {int(y_pt_norm)})** or **{report_ra}**")


# DESCRIPTION
header = st.container()
with header:
    st.title('Practice Risk Assessment Tool')
    st.markdown('Answer the questions in the sidebar by selecting options and sliding the bar to the appropriate \
    position. There are two sections to complete. If a question does not apply to your practice, answer "N/A".')
    st.markdown('The risk assessment score is calculated by dividing the sum of each section by the number \
    of questions answered. The raw score is plotted. The EGBC Guide indicates that a whole number score \
    can be used for reporting. The reported risk assessment category is based on the raw score.')
    st.markdown('Email mbuki.messi@gmail.com for any feedback or inquiries.')
    st.divider()