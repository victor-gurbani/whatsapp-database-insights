# 💬 WhatsApp Database Insights

A powerful Streamlit-based analyzer for your WhatsApp message database. Get deep insights into your chat patterns, behavioral analytics, gender distribution, and more.

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/streamlit-1.0+-FF4B4B.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

## 🚀 Quickstart

Run this single command to clone, setup, and launch the app:

```bash
curl -sSL https://raw.githubusercontent.com/victor-gurbani/whatsapp-database-insights/main/quickstart.sh | bash
```

Or if you prefer `wget`:

```bash
wget -qO- https://raw.githubusercontent.com/victor-gurbani/whatsapp-database-insights/main/quickstart.sh | bash
```

> **Note**: This works on **macOS**, **Linux**, and **Windows** (via Git Bash or WSL).

---

## 📋 Manual Setup

If you prefer to set things up manually:

```bash
# 1. Clone the repository
git clone https://github.com/victor-gurbani/whatsapp-database-insights.git
cd whatsapp-database-insights

# 2. Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r wa_analyzer/requirements.txt

# 4. Run the app
streamlit run app.py
```

---

## 📱 Getting Your WhatsApp Database

To use this analyzer, you need to extract your WhatsApp database files from your Android device:

### Required Files
- `msgstore.db` - Main message database
- `wa.db` - Contacts database  
- `contacts.vcf` (optional) - Exported contacts for better name resolution

### Extraction Methods
1. **Root access**: Copy directly from `/data/data/com.whatsapp/databases/`
2. **Backup decryption**: Use tools like [WhatsApp-Backup-Downloader-Decryptor](https://github.com/giacomoferretti/whatsapp-backup-downloader-decryptor/issues)
3. **ADB backup**: For older Android versions

Place the `.db` files in the project root folder before running the app.

---

## ✨ Features

- 📊 **Activity Analytics** - Message volume, hourly patterns, top talkers
- 🔥 **Behavioral Patterns** - Ghosting stats, reply times, conversation initiators
- 👫 **Gender Insights** - Demographics breakdown with advanced metrics
- 📝 **Word Clouds** - Visual representation of your most used words
- 🔍 **Chat Explorer** - Deep dive into individual conversations
- 🎪 **Fun Insights** - Discover interesting patterns in your chats
- 🗺️ **Map View** - Visualize location-based data

---

## 🔧 Requirements

- Python 3.8+
- Git
- ~500MB disk space for dependencies

---

## 📄 License

MIT License - See [LICENSE](LICENSE) for details.

---

## 🤝 Contributing

Pull requests are welcome! For major changes, please open an issue first to discuss what you would like to change.

---

Made with ❤️ for WhatsApp data enthusiasts
