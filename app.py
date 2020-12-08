from io import StringIO
import streamlit as st
import numpy as np
import pandas as pd
import json
from matplotlib import pyplot as plt
import seaborn as sns
from datetime import datetime

IEEE_REG_PATH = './regions.json'
FIELDS_PATH = './fields.json'

with open(FIELDS_PATH, 'r') as fields_file:
    COLUMNS = json.load(fields_file)

with open(IEEE_REG_PATH, 'r') as reg_file:
    REGIONS = json.load(reg_file)

REPORT_FILE = st.sidebar.file_uploader("Upload TNSM report")

country_region = {v: k for k, values in REGIONS.items() for v in values}

def get_status_category(r):
    if pd.notnull(r['Manuscript Status']):
        if 'Accept' in r['Manuscript Status']:
            return 'Accept'
        elif 'Reject' in r['Manuscript Status']:
            return 'Reject'
        elif 'Withdrawn' in r['Manuscript Status']:
            return 'Withdrawn'
        # Papers which have not been withdrawn, for which the status is not 'Reject & Resubmit' and no final decision has been made are still in process
        elif 'Withdrawn' not in r['Manuscript Status'] and 'Reject & Resubmit' != r['Manuscript Status'] and not pd.notnull(r['Accept or Reject Final Decision']) and pd.notnull(r['Revised']):
            return 'Still in process'
        else:
            return np.nan
    else: # Papers without Manuscript status are still in process
        return 'Still in process'

def filter_submitted_per_year(year, count_type='total'): # count_type: ['original', 'revised', 'total']
    if count_type == 'total':
        return report_df[report_df['Submission Year'] == year].copy()
    elif count_type == 'original':
        return report_df[(report_df['Submission Year'] == year) & (report_df['Revised'] == False)].copy()
    else:
        return report_df[(report_df['Submission Year'] == year) & (report_df['Revised'] == True)].copy()

def filter_submitted_per_date_range(from_date, to_date):
    return report_df[(report_df['Original Submission Date'] >= from_date) & (report_df['Original Submission Date'] < to_date)].copy() 


def count_submitted_per_year(year, count_type='total'):
    return filter_submitted_per_year(year, count_type).shape[0]

def count_submitted_per_date_range(from_date, to_date):
    return filter_submitted_per_date_range(from_date, to_date).shape[0]

def redefine_status(r):
    if pd.notnull(r['Manuscript Status']):
        if 'Withdrawn' not in r['Manuscript Status'] and 'Reject & Resubmit' != r['Manuscript Status'] and not pd.notnull(r['Accept or Reject Final Decision']) and pd.notnull(r['Revised']):
            return 'Still in process'
        else:
            return r['Manuscript Status']
    else: # Papers without Manuscript status are still in process
        return 'Still in process'

st.title('IEEE TNSM Stats')

if REPORT_FILE is not None:
    report_df = pd.read_excel(REPORT_FILE)
    # test if the required columns are present in the provide report
    assert set(report_df.columns) == set(COLUMNS)

    report_df = report_df[report_df['Manuscript ID - Original'] != 'draft']
    report_df = report_df[report_df['Manuscript ID - Latest'] != 'draft']
    st.header('Data Preview')
    st.dataframe(report_df.head(5))
    
    report_df['First Decision Month Number'] = pd.DatetimeIndex(report_df['First Decision Date']).month
    report_df['First Decision Year'] = pd.DatetimeIndex(report_df['First Decision Date']).year
    report_df['Latest Decision Month Number'] = pd.DatetimeIndex(report_df['Latest Decision Date']).month
    report_df['Latest Decision Year'] = pd.DatetimeIndex(report_df['Latest Decision Date']).year
    report_df['Revised'] = report_df.apply(lambda r: '.R' in r['Manuscript ID - Latest'] if type(r['Manuscript ID - Latest']) != float else np.nan , axis=1)
    report_df['Current Status Category'] = report_df.apply(get_status_category, axis=1)
    report_df['Region'] = report_df['Contact Author Country/Region'].apply(lambda k: country_region[k] if pd.notnull(k) else np.nan)

    status_count_df = report_df.groupby(['Current Status Category', 'Submission Year']).size().unstack(fill_value=0)
    status_count_df.loc["Submitted"] = status_count_df.sum()
    status_count_df.loc["Acceptance rate"] = (status_count_df.loc['Accept'] / status_count_df.loc['Submitted']) * 100
    st.header('Paper status at a glance')
    st.table(status_count_df.round(2))

    accept_subms_df = pd.pivot_table(report_df[report_df['Accept or Reject Final Decision'] == 'Accept'], values='Manuscript ID - Original', index=['Latest Decision Year'], columns=['Submission Year'], aggfunc='count', margins=True).fillna('')
    st.header('Accepted papers')
    st.write('Year of submission (columns) vs. Year of acceptance (Rows)')
    st.table(accept_subms_df)

    last_two_year_df = report_df[report_df['Submission Year'].max() - report_df['Submission Year'] < 3].copy()
    last_two_year_df['Manuscript Status'] = last_two_year_df.apply(redefine_status, axis=1)
    last_two_year_stats_df = last_two_year_df.groupby(['Manuscript Status', 'Submission Year']).size().unstack(fill_value=0)
    last_two_year_stats_df.loc['Total processed (%)'] = (((last_two_year_stats_df.sum() - last_two_year_stats_df.loc['Still in process'])/last_two_year_stats_df.sum()) * 100).round(2)
    st.header('Detailed accept/reject number')
    st.write('Stats for the last 3 years')
    st.table(last_two_year_stats_df)

    st.header('Distribution of the number of days from first submission to first decision')
    fig0, ax0 = plt.subplots(figsize=(8,6))
    sns.ecdfplot(data=report_df[report_df['Submission Year'].max() - report_df['Submission Year'] < 5], x="# Days Between Original Submission & Original Decision", hue="Submission Year", ax=ax0)
    ax0.set_title('CDF: # Days Between Original Submission & Original Decision')
    ax0.set_ylabel('CDF')
    st.pyplot(fig0)

    st.header('Distribution of the number of days from first submission to final decision')
    fig1, ax1 = plt.subplots(figsize=(8,6))
    sns.ecdfplot(data=report_df[report_df['Submission Year'].max() - report_df['Submission Year'] < 5], x="# Days Between Original Submission & Final Decision", hue="Submission Year", ax=ax1)
    ax1.set_title('CDF: # Days Between Original Submission & Final Decision')
    ax1.set_ylabel('CDF')
    st.pyplot(fig1)
    
    st.header('Geographical distribution of submissions')
    subms_region_df = report_df.groupby(['Region', 'Submission Year']).size().unstack(fill_value=0)
    subms_region_df = ((subms_region_df / subms_region_df.sum())*100)
    st.table(subms_region_df.round(2))

    st.header('Geographical distribution of publications')
    accept_region_df = report_df[report_df['Accept or Reject Final Decision'] == 'Accept'].groupby(['Region', 'Latest Decision Year']).size().unstack(fill_value=0)
    accept_region_df = ((accept_region_df / accept_region_df.sum())*100)
    st.table(accept_region_df.round(2))

    st.header('Longest first review round')
    # Longest First Review
    days_original_decision_max_idx = [idx[1] for idx in report_df.loc[report_df['Revised'] == False, :].groupby(['Submission Year'])['# Days Between Original Submission & Original Decision'].nlargest(1).index]
    max_original_decision_df = report_df[['Submission Year', '# Days Between Original Submission & Original Decision', 'Manuscript ID - Latest']].loc[days_original_decision_max_idx].set_index('Submission Year')
    st.table(max_original_decision_df)

    st.header('Longest second+ review round')
    # Longest Second+ Review
    days_second_review_max_idx = [idx[1] for idx in report_df.loc[report_df['Revised'] == True, :].groupby(['Submission Year'])['# Days Between Original Submission & Original Decision'].nlargest(1).index]
    max_second_review_df = report_df[['Submission Year', '# Days Between Original Submission & Original Decision', 'Manuscript ID - Latest']].loc[days_second_review_max_idx].set_index('Submission Year')
    st.table(max_second_review_df)
    
    st.header('Longest review round overall (from submission to final decision)')
    # Longest Overall Review (from submission to final decision)
    days_final_decision_max_idx = [idx[1] for idx in report_df.groupby(['Submission Year'])['# Days Between Original Submission & Final Decision'].nlargest(1).index]
    max_final_decision_df = report_df[['Submission Year', '# Days Between Original Submission & Final Decision', 'Manuscript ID - Latest']].loc[days_final_decision_max_idx].set_index('Submission Year')
    st.table(max_final_decision_df)
else:
    st.markdown('No info available yet (*Use the file upload input in the sidebar*)')






