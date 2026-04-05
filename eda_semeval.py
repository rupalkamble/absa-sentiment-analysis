# ============================================================
# Aspect-Based Sentiment Analysis — EDA on SemEval Datasets
# ============================================================
# What is EDA?
#   Exploratory Data Analysis means we look at the dataset
#   carefully BEFORE building any model. We want to understand:
#     - How many reviews are there?
#     - What aspects are being talked about?
#     - Are sentiments balanced or skewed?
#     - How long are the sentences?
# ============================================================

# --- STEP 1: Install required libraries (run once in terminal) ---
# pip install pandas matplotlib seaborn lxml nltk wordcloud

import os
import xml.etree.ElementTree as ET   # SemEval files are in XML format
import pandas as pd                  # For tables/dataframes
import matplotlib.pyplot as plt      # For charts
import seaborn as sns                # For prettier charts
from collections import Counter      # Counts occurrences easily
from wordcloud import WordCloud      # Makes word cloud images
import nltk
nltk.download('punkt', quiet=True)


# ============================================================
# STEP 2: Parse the SemEval XML files into a DataFrame
# ============================================================
# SemEval 2014/2016 data looks like this in XML:
#
#   <sentence id="...">
#     <text>The pizza was great but service was slow.</text>
#     <aspectTerms>
#       <aspectTerm term="pizza" polarity="positive" .../>
#       <aspectTerm term="service" polarity="negative" .../>
#     </aspectTerms>
#   </sentence>
#
# We'll extract each aspect term as one row in our table.

def parse_semeval_xml(filepath):
    """
    Reads a SemEval XML file and returns a pandas DataFrame.
    Each row = one aspect mention in one sentence.
    """
    tree = ET.parse(filepath)
    root = tree.getroot()

    records = []
    for sentence in root.findall('.//sentence'):
        sentence_id = sentence.get('id', '')
        text_el = sentence.find('text')
        text = text_el.text if text_el is not None else ''

        # Some files use 'aspectTerms', some use 'Opinions' (2016)
        for container_tag in ['aspectTerms', 'Opinions']:
            container = sentence.find(container_tag)
            if container is not None:
                child_tag = 'aspectTerm' if container_tag == 'aspectTerms' else 'Opinion'
                for aspect in container.findall(child_tag):
                    term     = aspect.get('term', aspect.get('target', 'NULL'))
                    polarity = aspect.get('polarity', aspect.get('sentiment', 'unknown'))
                    category = aspect.get('category', 'unknown')
                    from_idx = aspect.get('from', '-1')
                    to_idx   = aspect.get('to', '-1')

                    records.append({
                        'sentence_id': sentence_id,
                        'text':        text,
                        'aspect_term': term,
                        'polarity':    polarity,
                        'category':    category,
                        'from':        int(from_idx),
                        'to':          int(to_idx),
                        'text_length': len(text.split()),  # word count
                    })

    df = pd.DataFrame(records)
    return df


# ============================================================
# STEP 3: Load your datasets
# ============================================================
# Download SemEval 2014 data from:
#   https://alt.qcri.org/semeval2014/task4/
# Place the XML files in a folder called 'data/'
#
# Expected file names (adjust if yours differ):
DATA_FILES = {
    'Restaurant 2014 Train': 'data/Restaurants_Train_v2.xml',
    'Laptop 2014 Train':     'data/Laptop_Train_v2.xml',
    'Restaurant 2016 Train': 'data/ABSA16_Restaurants_Train_SB1_v2.xml',
}

# Auto-search: if file not found at expected path, search inside subfolders of data/
def find_file(expected_path):
    if os.path.exists(expected_path):
        return expected_path
    filename = os.path.basename(expected_path)
    for root, dirs, files in os.walk('data'):
        for f in files:
            if f == filename:
                found = os.path.join(root, f)
                print(f"   📂 Found at: {found}")
                return found
    return None

dataframes = {}
for name, path in DATA_FILES.items():
    found_path = find_file(path)
    if found_path:
        df = parse_semeval_xml(found_path)
        df['dataset'] = name
        if df.empty:
            print(f"⚠️  '{name}' parsed but contained no aspect records — skipping")
        else:
            dataframes[name] = df
            print(f"✅ Loaded '{name}': {len(df)} aspect records from {df['sentence_id'].nunique()} sentences")
    else:
        print(f"⚠️  File not found: {path}  — skipping '{name}'")

# Combine all into one master DataFrame
if dataframes:
    master_df = pd.concat(dataframes.values(), ignore_index=True)
else:
    # If no files found, create a small fake dataset so you can still run the code
    print("\n📌 No XML files found. Using a tiny demo dataset so the code still runs.\n")
    master_df = pd.DataFrame({
        'sentence_id': ['s1','s1','s2','s2','s3','s3','s4'],
        'text': [
            'The pizza was amazing but the service was terrible.',
            'The pizza was amazing but the service was terrible.',
            'Great ambiance, average food.',
            'Great ambiance, average food.',
            'Battery life is excellent, keyboard feels cheap.',
            'Battery life is excellent, keyboard feels cheap.',
            'Absolutely loved the pasta!',
        ],
        'aspect_term': ['pizza','service','ambiance','food','battery life','keyboard','pasta'],
        'polarity':    ['positive','negative','positive','neutral','positive','negative','positive'],
        'category':    ['FOOD','SERVICE','AMBIENCE','FOOD','BATTERY','HARDWARE','FOOD'],
        'from':        [4,29,6,16,0,18,16],
        'to':          [9,36,14,20,12,26,21],
        'text_length': [10,10,5,5,9,9,4],
        'dataset':     ['Restaurant 2014 Train']*4 + ['Laptop 2014 Train']*2 + ['Restaurant 2014 Train'],
    })


# ============================================================
# STEP 4: Basic statistics — always print these first!
# ============================================================
print("\n" + "="*55)
print("BASIC STATISTICS")
print("="*55)
print(f"Total aspect records : {len(master_df)}")
print(f"Unique sentences     : {master_df['sentence_id'].nunique()}")
print(f"Avg aspects/sentence : {len(master_df)/master_df['sentence_id'].nunique():.2f}")
print(f"\nPolarity distribution:\n{master_df['polarity'].value_counts()}")
print(f"\nDataset sizes:\n{master_df['dataset'].value_counts()}")

# Show first 5 rows so you can see what the data looks like
print("\nSample rows:")
print(master_df[['text','aspect_term','polarity','category']].head())


# ============================================================
# STEP 5: Visualizations — 6 plots saved to one figure
# ============================================================
# We use a 2x3 grid of subplots (2 rows, 3 columns)

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle('SemEval ABSA — Exploratory Data Analysis', fontsize=16, fontweight='bold')

# --- Plot 1: Sentiment distribution (bar chart) ---
ax = axes[0, 0]
polarity_counts = master_df['polarity'].value_counts()
colors = {'positive': '#2ecc71', 'negative': '#e74c3c', 'neutral': '#3498db', 'conflict': '#f39c12'}
bar_colors = [colors.get(p, '#95a5a6') for p in polarity_counts.index]
polarity_counts.plot(kind='bar', ax=ax, color=bar_colors, edgecolor='black', width=0.6)
ax.set_title('Sentiment Class Distribution')
ax.set_xlabel('Polarity')
ax.set_ylabel('Count')
ax.tick_params(axis='x', rotation=0)
for bar in ax.patches:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
            str(int(bar.get_height())), ha='center', fontsize=10)

# --- Plot 2: Sentiments per dataset (grouped bar) ---
ax = axes[0, 1]
pivot = master_df.groupby(['dataset', 'polarity']).size().unstack(fill_value=0)
pivot.plot(kind='bar', ax=ax, edgecolor='black', width=0.7,
           color=[colors.get(c, '#95a5a6') for c in pivot.columns])
ax.set_title('Sentiment Distribution per Dataset')
ax.set_xlabel('')
ax.set_ylabel('Count')
ax.tick_params(axis='x', rotation=20)
ax.legend(title='Polarity', fontsize=8)

# --- Plot 3: Top 15 most common aspect terms ---
ax = axes[0, 2]
# Exclude 'NULL' (means the aspect has no explicit term)
aspect_counts = master_df[master_df['aspect_term'] != 'NULL']['aspect_term'].value_counts().head(15)
aspect_counts.plot(kind='barh', ax=ax, color='#5dade2', edgecolor='black')
ax.set_title('Top 15 Aspect Terms')
ax.set_xlabel('Count')
ax.invert_yaxis()  # Most common on top

# --- Plot 4: Text length distribution (histogram) ---
ax = axes[1, 0]
for dataset_name, group in master_df.groupby('dataset'):
    ax.hist(group['text_length'], bins=20, alpha=0.6, label=dataset_name, edgecolor='black')
ax.set_title('Sentence Length Distribution (words)')
ax.set_xlabel('Word Count')
ax.set_ylabel('Frequency')
ax.legend(fontsize=7)

# --- Plot 5: Aspects per sentence distribution ---
ax = axes[1, 1]
aspects_per_sentence = master_df.groupby('sentence_id').size()
ax.hist(aspects_per_sentence, bins=range(1, aspects_per_sentence.max()+2),
        color='#a29bfe', edgecolor='black', align='left')
ax.set_title('Number of Aspects per Sentence')
ax.set_xlabel('# Aspects')
ax.set_ylabel('# Sentences')
# Annotate: sentences with >1 aspect are "multi-aspect" — important for ABSA!
multi_aspect_pct = (aspects_per_sentence > 1).mean() * 100
ax.text(0.6, 0.85, f'Multi-aspect:\n{multi_aspect_pct:.1f}% of sentences',
        transform=ax.transAxes, fontsize=10,
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

# --- Plot 6: Top 10 aspect categories ---
ax = axes[1, 2]
cat_counts = master_df[master_df['category'] != 'unknown']['category'].value_counts().head(10)
if len(cat_counts) > 0:
    cat_counts.plot(kind='barh', ax=ax, color='#fd79a8', edgecolor='black')
    ax.set_title('Top 10 Aspect Categories')
    ax.set_xlabel('Count')
    ax.invert_yaxis()
else:
    ax.text(0.5, 0.5, 'No category data\nin this dataset',
            ha='center', va='center', transform=ax.transAxes, fontsize=12)
    ax.set_title('Aspect Categories')

plt.tight_layout()
plt.savefig('eda_plots.png', dpi=150, bbox_inches='tight')
plt.show()
print("\n✅ Plots saved to 'eda_plots.png'")


# ============================================================
# STEP 6: Word Cloud — visual of most frequent words
# ============================================================
# Separate words by sentiment so we can see which words
# are associated with positive vs negative reviews.

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('Word Clouds by Sentiment', fontsize=14, fontweight='bold')

wc_colors = {'positive': 'Greens', 'negative': 'Reds', 'neutral': 'Blues'}

for ax, polarity in zip(axes, ['positive', 'negative', 'neutral']):
    subset = master_df[master_df['polarity'] == polarity]
    text_blob = ' '.join(subset['text'].dropna().tolist())

    if text_blob.strip():
        wc = WordCloud(
            width=500, height=300,
            background_color='white',
            colormap=wc_colors.get(polarity, 'viridis'),
            max_words=80,
        ).generate(text_blob)
        ax.imshow(wc, interpolation='bilinear')
    else:
        ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)

    ax.axis('off')
    ax.set_title(f'{polarity.capitalize()} Reviews', fontsize=12)

plt.tight_layout()
plt.savefig('wordclouds.png', dpi=150, bbox_inches='tight')
plt.show()
print("✅ Word clouds saved to 'wordclouds.png'")


# ============================================================
# STEP 7: Conflict detection — sentences with mixed sentiment
# ============================================================
# A sentence like "Pizza was great but service was terrible"
# has BOTH positive and negative aspects. These are tricky
# for models — let's count how many exist.

sentence_sentiments = master_df.groupby('sentence_id')['polarity'].apply(set)
conflicting = sentence_sentiments[sentence_sentiments.apply(
    lambda s: 'positive' in s and 'negative' in s
)]

print(f"\n⚡ Sentences with conflicting sentiments: {len(conflicting)}")
print(f"   That's {len(conflicting)/master_df['sentence_id'].nunique()*100:.1f}% of all sentences")
print("   These are the hardest cases for any ABSA model!\n")

# Show a few examples
example_ids = list(conflicting.index[:3])
if example_ids:
    print("Example conflicting sentences:")
    examples = master_df[master_df['sentence_id'].isin(example_ids)][
        ['sentence_id','text','aspect_term','polarity']
    ].drop_duplicates()
    print(examples.to_string(index=False))


# ============================================================
# STEP 8: Save a summary CSV for your report
# ============================================================
summary = master_df.groupby(['dataset','polarity']).size().reset_index(name='count')
summary.to_csv('eda_summary.csv', index=False)
print("\n✅ Summary CSV saved to 'eda_summary.csv'")
print("\n🎉 EDA complete! Next step: BIO Tagging for Aspect Extraction.")
