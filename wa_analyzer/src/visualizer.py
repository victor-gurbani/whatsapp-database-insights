import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import os
import matplotlib.dates as mdates

class ReportGenerator:
    def __init__(self, output_dir):
        self.output_dir = output_dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        sns.set_theme(style="whitegrid")

    def plot_top_talkers(self, top_talkers):
        if top_talkers is None:
            return

        # Accept either:
        # - Series indexed by contact
        # - DataFrame with contact/count columns (current analyzer output)
        if hasattr(top_talkers, "reset_index") and not hasattr(top_talkers, "columns"):
            plot_df = top_talkers.reset_index()
            plot_df.columns = ["contact_name", "count"]
        elif hasattr(top_talkers, "columns"):
            plot_df = top_talkers.copy()
            if "contact_name" not in plot_df.columns and "chat_name" in plot_df.columns:
                plot_df = plot_df.rename(columns={"chat_name": "contact_name"})
            if "count" not in plot_df.columns:
                return
            plot_df = plot_df[["contact_name", "count"]]
        else:
            return

        if plot_df.empty:
            return

        plt.figure(figsize=(10, 6))
        sns.barplot(data=plot_df, x="count", y="contact_name", palette="viridis", hue="contact_name", legend=False)
        plt.title("Top 10 Most Active Contacts")
        plt.xlabel("Number of Messages")
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "top_talkers.png"))
        plt.close()

    def plot_hourly_activity(self, hourly_stats):
        plt.figure(figsize=(10, 4))
        sns.lineplot(x=hourly_stats.index, y=hourly_stats.values, marker="o", color="coral")
        plt.title("Activity by Hour of Day")
        plt.xlabel("Hour (0-23)")
        plt.ylabel("Message Count")
        plt.xticks(range(0, 24))
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "hourly_activity.png"))
        plt.close()

    def plot_timeline(self, monthly_stats):
        plt.figure(figsize=(12, 6))
        
        # monthly_stats index is DatetimeIndex
        plt.plot(monthly_stats.index, monthly_stats.values, label='Messages')
        
        # Format x-axis
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        plt.gcf().autofmt_xdate()
        
        plt.title("Message Volume Over Time")
        plt.ylabel("Count")
        plt.legend()
        plt.savefig(os.path.join(self.output_dir, "timeline.png"))
        plt.close()

    def plot_gender_stats(self, gender_counts):
        plt.figure(figsize=(6, 6))
        data = gender_counts[gender_counts.index != 'unknown'] # Exclude unknown for cleaner pie
        if data.empty:
            data = gender_counts # Fallback
            
        plt.pie(data.values, labels=data.index, autopct='%1.1f%%', colors=sns.color_palette("pastel"))
        plt.title("Messages by Contact Gender")
        plt.savefig(os.path.join(self.output_dir, "gender_stats.png"))
        plt.close()

    def generate_wordcloud(self, text, filename="wordcloud.png"):
        if not text:
            return
        wc = WordCloud(width=800, height=400, background_color='white').generate(text)
        plt.figure(figsize=(10, 5))
        plt.imshow(wc, interpolation='bilinear')
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, filename))
        plt.close()

    def generate_report_md(self, stats, simple_stats):
        md = f"""# WhatsApp Analysis Report

## Basic Statistics
- **Total Messages**: {simple_stats.get('total_messages')}
- **Messages Sent**: {simple_stats.get('sent_count')}
- **Messages Received**: {simple_stats.get('received_count')}
- **Unique Contacts**: {simple_stats.get('unique_contacts')}
- **Average Reply Time (Sent)**: {simple_stats.get('avg_reply_minutes', 0):.2f} minutes

## Top Contacts
![Top Talkers](top_talkers.png)

## Activity Patterns
![Hourly Activity](hourly_activity.png)

![Timeline](timeline.png)

## Demographics (Gender Estimation)
> Note: Based on first name heuristics.
![Gender Stats](gender_stats.png)

## Word Cloud (Overall)
![Word Cloud](wordcloud.png)
"""
        with open(os.path.join(self.output_dir, "report.md"), "w") as f:
            f.write(md)
