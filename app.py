import os
import gradio as gr
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from groq import Groq

MENU = [{'id': 'esp01', 'name': 'Classic Espresso', 'category': 'Coffee - Hot', 'tags': 'strong bitter quick caffeine kick no milk', 'price': 120, 'desc': 'Double shot espresso, bold and intense.'}, {'id': 'lat01', 'name': 'Vanilla Latte', 'category': 'Coffee - Hot', 'tags': 'sweet creamy milky mild vanilla comfort', 'price': 180, 'desc': 'Espresso with steamed milk and vanilla syrup.'}, {'id': 'cap01', 'name': 'Cappuccino', 'category': 'Coffee - Hot', 'tags': 'foamy creamy classic balanced milk', 'price': 160, 'desc': 'Espresso, steamed milk, thick foam.'}, {'id': 'mocha01', 'name': 'Dark Mocha', 'category': 'Coffee - Hot', 'tags': 'chocolate rich indulgent dessert sweet', 'price': 200, 'desc': 'Espresso, dark chocolate, steamed milk.'}, {'id': 'cold01', 'name': 'Cold Brew', 'category': 'Coffee - Cold', 'tags': 'smooth strong refreshing summer low acid no sugar', 'price': 190, 'desc': 'Slow-steeped 18hrs, smooth and strong, served over ice.'}, {'id': 'iced01', 'name': 'Iced Caramel Macchiato', 'category': 'Coffee - Cold', 'tags': 'sweet caramel refreshing summer indulgent', 'price': 210, 'desc': 'Espresso, milk, vanilla, caramel drizzle, iced.'}, {'id': 'matcha01', 'name': 'Iced Matcha Latte', 'category': 'Tea/Other - Cold', 'tags': 'healthy no caffeine crash antioxidant earthy refreshing', 'price': 220, 'desc': 'Ceremonial grade matcha, milk, ice.'}, {'id': 'chai01', 'name': 'Masala Chai', 'category': 'Tea - Hot', 'tags': 'spiced comforting mild caffeine indian classic', 'price': 100, 'desc': 'Traditional spiced tea with milk.'}]

CUSTOMER_HISTORY = {'priya': ['Vanilla Latte', 'Iced Caramel Macchiato'], 'arjun': ['Classic Espresso', 'Cold Brew'], 'guest': []}

client = Groq(api_key=os.environ["GROQ_API_KEY"])

texts = [f"{m[\'name\']} {m[\'category\']} {m[\'tags\']} {m[\'desc\']}" for m in MENU]
vectorizer = TfidfVectorizer(stop_words="english")
menu_matrix = vectorizer.fit_transform(texts)

def retrieve(query, top_k=4):
    qv = vectorizer.transform([query])
    sims = cosine_similarity(qv, menu_matrix).flatten()
    idx = sims.argsort()[::-1][:top_k]
    return [MENU[i] for i in idx if sims[i] > 0] or MENU[:top_k]

SYSTEM_PROMPT = """You are Bean Buddy, a friendly AI barista. Recommend ONLY from
the retrieved menu items given — never invent items/prices. Be warm, concise
(3-5 sentences). End with: "Top pick: <item name> - ₹<price>"."""

def ask_agent(customer, query):
    history = CUSTOMER_HISTORY.get(customer.lower(), [])
    retrieved = retrieve(f"{query} " + " ".join(history))
    menu_block = "\n".join(f"- {m[\'name\']} (₹{m[\'price\']}): {m[\'desc\']}" for m in retrieved)
    user_prompt = f"Customer: {customer}\nPast orders: {history}\nThey say: \"{query}\"\n\nMenu options:\n{menu_block}"
    resp = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[{"role": "system", "content": SYSTEM_PROMPT},
                   {"role": "user", "content": user_prompt}],
        temperature=0.6, max_tokens=600, reasoning_effort="low",
    )
    return resp.choices[0].message.content

with gr.Blocks(title="Bean Buddy") as demo:
    gr.Markdown("## ☕ Bean Buddy — AI Coffee Agent")
    customer = gr.Textbox(label="Customer name", value="priya")
    gr.ChatInterface(fn=lambda msg, hist: ask_agent(customer.value, msg))

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)))
