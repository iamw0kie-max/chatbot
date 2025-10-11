import mysql.connector
from mysql.connector import Error
import PyPDF2
import streamlit as st
from datetime import datetime
import io


class PDFChatbotDB:
    def __init__(self, host='localhost', user='root', password='your_password', database='chatbot_db'):
        """Initialize database connection"""
        self.connection = None
        try:
            self.connection = mysql.connector.connect(
                host=host,
                user=user,
                password=password,
                database=database
            )
            if self.connection.is_connected():
                self.create_tables()
        except Error as e:
            st.error(f"Error connecting to MySQL: {e}")

    def create_tables(self):
        """Create necessary tables if they don't exist"""
        cursor = self.connection.cursor()

        cursor.execute('''
                       CREATE TABLE IF NOT EXISTS pdf_documents
                       (
                           id
                           INT
                           AUTO_INCREMENT
                           PRIMARY
                           KEY,
                           filename
                           VARCHAR
                       (
                           255
                       ) NOT NULL,
                           upload_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                           total_pages INT,
                           file_size INT
                           )
                       ''')

        cursor.execute('''
                       CREATE TABLE IF NOT EXISTS pdf_content
                       (
                           id
                           INT
                           AUTO_INCREMENT
                           PRIMARY
                           KEY,
                           document_id
                           INT,
                           page_number
                           INT,
                           content
                           TEXT,
                           FOREIGN
                           KEY
                       (
                           document_id
                       ) REFERENCES pdf_documents
                       (
                           id
                       ) ON DELETE CASCADE
                           )
                       ''')

        cursor.execute('''
                       CREATE TABLE IF NOT EXISTS chat_history
                       (
                           id
                           INT
                           AUTO_INCREMENT
                           PRIMARY
                           KEY,
                           document_id
                           INT,
                           user_message
                           TEXT,
                           bot_response
                           TEXT,
                           timestamp
                           TIMESTAMP
                           DEFAULT
                           CURRENT_TIMESTAMP,
                           FOREIGN
                           KEY
                       (
                           document_id
                       ) REFERENCES pdf_documents
                       (
                           id
                       ) ON DELETE CASCADE
                           )
                       ''')

        self.connection.commit()

    def extract_pdf_content(self, pdf_file):
        """Extract content from PDF file"""
        try:
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            num_pages = len(pdf_reader.pages)

            content = []
            for page_num in range(num_pages):
                page = pdf_reader.pages[page_num]
                text = page.extract_text()
                content.append({
                    'page_number': page_num + 1,
                    'text': text
                })

            return content, num_pages
        except Exception as e:
            st.error(f"Error extracting PDF content: {e}")
            return None, 0

    def upload_pdf(self, pdf_file, filename):
        """Upload PDF and store its content in database"""
        file_size = pdf_file.size

        # Extract PDF content
        content, num_pages = self.extract_pdf_content(pdf_file)
        if content is None:
            return None

        cursor = self.connection.cursor()

        # Insert document metadata
        cursor.execute('''
                       INSERT INTO pdf_documents (filename, total_pages, file_size)
                       VALUES (%s, %s, %s)
                       ''', (filename, num_pages, file_size))

        document_id = cursor.lastrowid

        # Insert content for each page
        for page_data in content:
            cursor.execute('''
                           INSERT INTO pdf_content (document_id, page_number, content)
                           VALUES (%s, %s, %s)
                           ''', (document_id, page_data['page_number'], page_data['text']))

        self.connection.commit()
        return document_id

    def search_content(self, document_id, query):
        """Search for query in document content"""
        cursor = self.connection.cursor(dictionary=True)

        cursor.execute('''
                       SELECT page_number, content
                       FROM pdf_content
                       WHERE document_id = %s
                         AND content LIKE %s
                       ''', (document_id, f'%{query}%'))

        results = cursor.fetchall()
        return results

    def get_page_content(self, document_id, page_number):
        """Get content of a specific page"""
        cursor = self.connection.cursor(dictionary=True)

        cursor.execute('''
                       SELECT content
                       FROM pdf_content
                       WHERE document_id = %s
                         AND page_number = %s
                       ''', (document_id, page_number))

        result = cursor.fetchone()
        return result['content'] if result else None

    def get_all_content(self, document_id):
        """Get all content from a document"""
        cursor = self.connection.cursor(dictionary=True)

        cursor.execute('''
                       SELECT page_number, content
                       FROM pdf_content
                       WHERE document_id = %s
                       ORDER BY page_number
                       ''', (document_id,))

        results = cursor.fetchall()
        return results

    def get_document_info(self, document_id):
        """Get document information"""
        cursor = self.connection.cursor(dictionary=True)

        cursor.execute('''
                       SELECT filename, total_pages, file_size, upload_date
                       FROM pdf_documents
                       WHERE id = %s
                       ''', (document_id,))

        return cursor.fetchone()

    def save_chat(self, document_id, user_message, bot_response):
        """Save chat interaction to database"""
        cursor = self.connection.cursor()

        cursor.execute('''
                       INSERT INTO chat_history (document_id, user_message, bot_response)
                       VALUES (%s, %s, %s)
                       ''', (document_id, user_message, bot_response))

        self.connection.commit()

    def get_chat_history(self, document_id):
        """Retrieve chat history"""
        cursor = self.connection.cursor(dictionary=True)

        cursor.execute('''
                       SELECT user_message, bot_response, timestamp
                       FROM chat_history
                       WHERE document_id = %s
                       ORDER BY timestamp ASC
                       ''', (document_id,))

        return cursor.fetchall()

    def list_documents(self):
        """List all uploaded documents"""
        cursor = self.connection.cursor(dictionary=True)

        cursor.execute('''
                       SELECT id, filename, upload_date, total_pages, file_size
                       FROM pdf_documents
                       ORDER BY upload_date DESC
                       ''')

        return cursor.fetchall()

    def delete_document(self, document_id):
        """Delete a document and its content"""
        cursor = self.connection.cursor()
        cursor.execute('DELETE FROM pdf_documents WHERE id = %s', (document_id,))
        self.connection.commit()

    def close(self):
        """Close database connection"""
        if self.connection and self.connection.is_connected():
            self.connection.close()


def process_query(db, document_id, query):
    """Process user query and generate response"""
    query_lower = query.lower()

    # Handle different types of queries
    if 'search' in query_lower or 'find' in query_lower:
        # Extract search term
        search_terms = query.split()
        search_term = ' '.join(
            [word for word in search_terms if word.lower() not in ['search', 'find', 'for', 'the', 'in']])

        results = db.search_content(document_id, search_term)
        if results:
            response = f"Found '{search_term}' in {len(results)} page(s):\n\n"
            for result in results[:3]:  # Limit to 3 results
                snippet = result['content'][:300].replace('\n', ' ')
                response += f"**Page {result['page_number']}:**\n{snippet}...\n\n"
            return response
        else:
            return f"No results found for '{search_term}'"

    elif 'page' in query_lower:
        # Extract page number
        words = query.split()
        for i, word in enumerate(words):
            if word.lower() == 'page' and i + 1 < len(words):
                try:
                    page_num = int(words[i + 1])
                    content = db.get_page_content(document_id, page_num)
                    if content:
                        return f"**Page {page_num} content:**\n\n{content}"
                    else:
                        return f"Page {page_num} not found"
                except ValueError:
                    pass
        return "Please specify a page number (e.g., 'show page 5')"

    elif 'summary' in query_lower or 'summarize' in query_lower:
        doc_info = db.get_document_info(document_id)
        contents = db.get_all_content(document_id)

        # Get first 500 characters from first few pages
        preview = ""
        for page in contents[:3]:
            preview += page['content'][:500] + "\n\n"

        return f"**Document Summary:**\n\n**Filename:** {doc_info['filename']}\n**Total Pages:** {doc_info['total_pages']}\n\n**Preview:**\n{preview[:1000]}..."

    elif 'how many' in query_lower and 'page' in query_lower:
        doc_info = db.get_document_info(document_id)
        return f"This document has **{doc_info['total_pages']}** pages."

    else:
        # General search in entire document
        results = db.search_content(document_id, query)
        if results:
            response = f"Found relevant content in {len(results)} page(s):\n\n"
            for result in results[:3]:
                snippet = result['content'][:300].replace('\n', ' ')
                response += f"**Page {result['page_number']}:**\n{snippet}...\n\n"
            return response
        else:
            return "I couldn't find relevant information. Try asking about:\n- Searching for specific terms\n- Getting content from a specific page\n- Document summary\n- Number of pages"


def main():
    st.set_page_config(page_title="PDF Chatbot Assistant", page_icon="📄", layout="wide")

    st.title("📄 PDF Chatbot Assistant")
    st.markdown("Upload PDF documents and chat with them using natural language!")

    # Initialize session state
    if 'db' not in st.session_state:
        st.session_state.db = PDFChatbotDB(
            host='localhost',
            user='root',
            password='Quickpwd@123',  # Change this
            database='chatbot_db'
        )

    if 'current_doc_id' not in st.session_state:
        st.session_state.current_doc_id = None

    if 'messages' not in st.session_state:
        st.session_state.messages = []

    # Sidebar
    with st.sidebar:
        st.header("📁 Document Management")

        # Upload PDF
        uploaded_file = st.file_uploader("Upload PDF", type=['pdf'])
        if uploaded_file is not None:
            if st.button("Process PDF"):
                with st.spinner("Processing PDF..."):
                    doc_id = st.session_state.db.upload_pdf(uploaded_file, uploaded_file.name)
                    if doc_id:
                        st.session_state.current_doc_id = doc_id
                        st.session_state.messages = []
                        st.success(f"PDF uploaded successfully! Document ID: {doc_id}")
                        st.rerun()

        st.divider()

        # List documents
        st.subheader("Available Documents")
        docs = st.session_state.db.list_documents()

        if docs:
            for doc in docs:
                col1, col2 = st.columns([3, 1])
                with col1:
                    if st.button(f"📄 {doc['filename']}", key=f"doc_{doc['id']}"):
                        st.session_state.current_doc_id = doc['id']
                        # Load chat history
                        history = st.session_state.db.get_chat_history(doc['id'])
                        st.session_state.messages = []
                        for chat in history:
                            st.session_state.messages.append({
                                "role": "user",
                                "content": chat['user_message']
                            })
                            st.session_state.messages.append({
                                "role": "assistant",
                                "content": chat['bot_response']
                            })
                        st.rerun()
                with col2:
                    if st.button("🗑️", key=f"del_{doc['id']}"):
                        st.session_state.db.delete_document(doc['id'])
                        if st.session_state.current_doc_id == doc['id']:
                            st.session_state.current_doc_id = None
                            st.session_state.messages = []
                        st.rerun()

                st.caption(f"Pages: {doc['total_pages']} | {doc['upload_date']}")
        else:
            st.info("No documents uploaded yet")

    # Main chat interface
    if st.session_state.current_doc_id:
        doc_info = st.session_state.db.get_document_info(st.session_state.current_doc_id)

        st.info(f"**Current Document:** {doc_info['filename']} | **Pages:** {doc_info['total_pages']}")

        # Display chat messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # Chat input
        if prompt := st.chat_input("Ask something about the document..."):
            # Add user message
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            # Generate response
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    response = process_query(st.session_state.db, st.session_state.current_doc_id, prompt)
                    st.markdown(response)

            # Save to database
            st.session_state.db.save_chat(st.session_state.current_doc_id, prompt, response)
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.rerun()

        # Example queries
        st.divider()
        st.markdown("**💡 Example queries:**")
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("📊 Get summary"):
                prompt = "Give me a summary of this document"
                st.session_state.messages.append({"role": "user", "content": prompt})
                response = process_query(st.session_state.db, st.session_state.current_doc_id, prompt)
                st.session_state.db.save_chat(st.session_state.current_doc_id, prompt, response)
                st.session_state.messages.append({"role": "assistant", "content": response})
                st.rerun()

        with col2:
            if st.button("📄 How many pages?"):
                prompt = "How many pages in this document?"
                st.session_state.messages.append({"role": "user", "content": prompt})
                response = process_query(st.session_state.db, st.session_state.current_doc_id, prompt)
                st.session_state.db.save_chat(st.session_state.current_doc_id, prompt, response)
                st.session_state.messages.append({"role": "assistant", "content": response})
                st.rerun()

        with col3:
            if st.button("📖 Show page 1"):
                prompt = "Show page 1"
                st.session_state.messages.append({"role": "user", "content": prompt})
                response = process_query(st.session_state.db, st.session_state.current_doc_id, prompt)
                st.session_state.db.save_chat(st.session_state.current_doc_id, prompt, response)
                st.session_state.messages.append({"role": "assistant", "content": response})
                st.rerun()

    else:
        st.info("👈 Please upload a PDF or select a document from the sidebar to start chatting!")

        st.markdown("""
        ### How to use:
        1. **Upload a PDF** using the file uploader in the sidebar
        2. **Click "Process PDF"** to extract and store the content
        3. **Start chatting** with the document using natural language

        ### What you can ask:
        - 🔍 "Search for [keyword]"
        - 📄 "Show page [number]"
        - 📊 "Give me a summary"
        - 🔢 "How many pages?"
        - 💬 Any question about the document content
        """)


if __name__ == "__main__":
    main()