"""
Complete Test Suite for Enhanced RAG Pipeline Flow
Tests: Document → Original Storage + Enhanced Chunks → Vector Store → Query → Context Retrieval → Enhanced Response
"""

import os
import json
import time
from dotenv import load_dotenv
from testRag import RAGPipeline
from blobStorage import upload_blob

load_dotenv()

def print_section(title: str):
    """Print formatted section header"""
    print("\n" + "="*60)
    print(f" {title} ")
    print("="*60)

def print_subsection(title: str):
    """Print formatted subsection header"""
    print(f"\n--- {title} ---")

class EnhancedRAGTester:
    def __init__(self, index_name: str = "test-enhanced-pipeline"):
        self.rag_pipeline = RAGPipeline(
            index_name=index_name, 
            llm_gpt5=True,
            chunk_size=800,  # Smaller chunks for better testing
            chunk_overlap=100
        )
        self.test_documents = []
        self.processed_docs = []

    def create_test_documents(self):
        """Step 1: Create sample documents for testing"""
        print_section("STEP 1: CREATING TEST DOCUMENTS")
        
        documents = [
            {
                "content": """
                Artificial Intelligence (AI) and Machine Learning (ML) have revolutionized numerous industries in recent years. 
                AI refers to the simulation of human intelligence processes by machines, especially computer systems. 
                These processes include learning, reasoning, and self-correction.

                Machine Learning is a subset of AI that provides systems the ability to automatically learn and improve 
                from experience without being explicitly programmed. ML focuses on the development of computer programs 
                that can access data and use it to learn for themselves.

                There are three main types of machine learning:
                1. Supervised Learning: Uses labeled datasets to train algorithms
                2. Unsupervised Learning: Finds hidden patterns in data without labels
                3. Reinforcement Learning: Learns through interaction with environment

                Deep Learning is a subset of machine learning that uses neural networks with three or more layers. 
                These neural networks attempt to simulate the behavior of the human brain, allowing machines to 
                "learn" from large amounts of data.

                Natural Language Processing (NLP) is another important field within AI that focuses on the interaction 
                between computers and humans using natural language. NLP combines computational linguistics with 
                statistical, machine learning, and deep learning models.

                Computer Vision is the field of AI that trains computers to interpret and understand the visual world. 
                Using digital images from cameras and videos and deep learning models, machines can accurately identify 
                and classify objects and react to what they "see."

                The applications of AI are vast and growing, including healthcare diagnostics, autonomous vehicles, 
                financial fraud detection, recommendation systems, voice assistants, and many more domains.
                """,
                "title": "AI_and_ML_Overview",
                "doc_id": "ai_ml_001"
            },
            {
                "content": """
                Cloud Computing has transformed how businesses operate and manage their IT infrastructure. 
                Cloud computing is the delivery of computing services including servers, storage, databases, 
                networking, software, analytics, and intelligence over the internet.

                The main cloud service models are:
                1. Infrastructure as a Service (IaaS): Provides virtualized computing resources
                2. Platform as a Service (PaaS): Offers hardware and software tools over the internet
                3. Software as a Service (SaaS): Delivers software applications over the internet

                Cloud deployment models include:
                - Public Cloud: Services offered over the public internet
                - Private Cloud: Cloud computing resources used exclusively by one business
                - Hybrid Cloud: Combines public and private clouds
                - Multi-cloud: Uses multiple cloud computing services

                Major cloud providers include Amazon Web Services (AWS), Microsoft Azure, Google Cloud Platform (GCP), 
                and IBM Cloud. Each provider offers a comprehensive suite of services including compute, storage, 
                databases, machine learning, analytics, and developer tools.

                Cloud computing benefits include cost reduction, scalability, flexibility, automatic software updates, 
                increased collaboration, and enhanced security. However, challenges include data security concerns, 
                downtime risks, vendor lock-in, and compliance issues.

                DevOps practices have been greatly enhanced by cloud computing, enabling continuous integration and 
                continuous deployment (CI/CD) pipelines. Container technologies like Docker and orchestration platforms 
                like Kubernetes have become integral to modern cloud-native application development.

                Edge computing is an emerging trend that brings computation and data storage closer to data sources, 
                reducing latency and improving performance for real-time applications.
                """,
                "title": "Cloud_Computing_Guide",
                "doc_id": "cloud_001"
            },
            {
                "content": """
                Cybersecurity is the practice of protecting systems, networks, and programs from digital attacks. 
                These cyberattacks are usually aimed at accessing, changing, or destroying sensitive information, 
                extorting money from users, or interrupting normal business processes.

                Common types of cybersecurity threats include:
                - Malware: Software designed to damage or gain unauthorized access to systems
                - Phishing: Fraudulent attempts to obtain sensitive information
                - Ransomware: Malicious software that encrypts files and demands payment
                - Social Engineering: Psychological manipulation to divulge confidential information
                - SQL Injection: Code injection technique targeting data-driven applications
                - Cross-Site Scripting (XSS): Injection of client-side scripts into web pages

                Essential cybersecurity practices include:
                1. Strong password policies and multi-factor authentication
                2. Regular software updates and patch management
                3. Network segmentation and access controls
                4. Employee training and security awareness programs
                5. Regular security assessments and penetration testing
                6. Incident response planning and disaster recovery

                Cybersecurity frameworks like NIST Cybersecurity Framework, ISO 27001, and SOC 2 provide structured 
                approaches to managing cybersecurity risks. These frameworks help organizations identify, protect, 
                detect, respond to, and recover from cybersecurity events.

                Emerging cybersecurity challenges include securing Internet of Things (IoT) devices, protecting 
                against AI-powered attacks, securing cloud environments, and addressing the cybersecurity skills gap.

                Zero Trust security model operates on the principle of "never trust, always verify" and requires 
                verification of every transaction and access request, regardless of location or user credentials.
                """,
                "title": "Cybersecurity_Essentials",
                "doc_id": "cyber_001"
            }
        ]

        for doc in documents:
            print(f"Created document: {doc['title']} ({len(doc['content'])} characters)")
        
        self.test_documents = documents
        return documents

    def upload_and_process_documents(self):
        """Step 2: Upload documents to blob storage and process through pipeline"""
        print_section("STEP 2: DOCUMENT UPLOAD AND PROCESSING")
        
        azure_blob_url = os.getenv("BLOB_SAS_URL")
        container_name = os.getenv("BLOB_CONTAINER_NAME")
        
        for doc in self.test_documents:
            print_subsection(f"Processing: {doc['title']}")
            
            # Upload to blob storage
            blob_filename = f"test_{doc['doc_id']}.txt"
            upload_success = upload_blob(doc["content"].encode('utf-8'), blob_filename)
            
            if not upload_success:
                print(f"❌ Failed to upload {doc['title']}")
                continue
                
            blob_url = f"{azure_blob_url}/{container_name}/{blob_filename}"
            print(f"✅ Uploaded to blob: {blob_filename}")
            
            # Process through enhanced pipeline
            try:
                start_time = time.time()
                metadata = self.rag_pipeline.process_and_index_document(
                    blob_url=blob_url,
                    file_type="txt",
                    doc_id=doc['doc_id']
                )
                processing_time = time.time() - start_time
                
                print(f"✅ Processed successfully:")
                print(f"   - Status: {metadata.status}")
                print(f"   - Chunks: {metadata.chunk_count}")
                print(f"   - Processing time: {processing_time:.2f}s")
                
                self.processed_docs.append({
                    'doc_id': doc['doc_id'],
                    'title': doc['title'],
                    'metadata': metadata,
                    'blob_url': blob_url
                })
                
            except Exception as e:
                print(f"❌ Processing failed: {str(e)}")

    def test_vector_storage_verification(self):
        """Step 3: Verify vector storage and chunk metadata"""
        print_section("STEP 3: VECTOR STORAGE VERIFICATION")
        
        for doc in self.processed_docs:
            print_subsection(f"Verifying: {doc['title']}")
            
            # Test basic similarity search
            results = self.rag_pipeline.similarity_search_with_context(
                query=doc['doc_id'], 
                k=3
            )
            
            print(f"Found {len(results)} chunks for {doc['title']}")
            
            for i, result in enumerate(results[:2]):  # Show first 2 chunks
                print(f"  Chunk {i+1}:")
                print(f"    - Content preview: {result['chunk_content'][:100]}...")
                print(f"    - Similarity score: {result['similarity_score']:.4f}")
                print(f"    - Document ID: {result['document_id']}")
                print(f"    - Chunk index: {result['chunk_index']}")
                print(f"    - Position: {result['start_char']}-{result['end_char']}")

    def test_enhanced_queries(self):
        """Step 4: Test enhanced query capabilities"""
        print_section("STEP 4: ENHANCED QUERY TESTING")
        
        test_queries = [
            {
                "query": "Who is Elon Musk?",
                "expected_doc": "AI_and_ML_Overview"
            },
        ]
        
        for i, test in enumerate(test_queries, 1):
            print_subsection(f"Query {i}: {test['query']}")
            
            try:
                start_time = time.time()
                
                # Test with context retrieval
                result = self.rag_pipeline.query(
                    query_text=test['query'],
                    top_k=3,
                    temperature=0.1,
                    include_context=True,
                    context_chars=200
                )
                
                query_time = time.time() - start_time
                
                print(f"✅ Query completed in {query_time:.2f}s")
                print(f"📝 Answer: {result['answer']}")
                print(f"📊 Sources found: {len(result['enhanced_sources'])}")
                print(f"🔧 Tokens used: {result['tokens_used']}")
                
                # Show enhanced source info
                for j, source in enumerate(result['enhanced_sources'][:2]):
                    print(f"   Source {j+1}:")
                    print(f"     - File: {source['filename']}")
                    print(f"     - Score: {source['similarity_score']:.4f}")
                    print(f"     - Chunk: {source['chunk_index']}")
                    
                    if source.get('document_context'):
                        context = source['document_context']
                        print(f"     - Context available: {len(context['full_context'])} chars")
                        print(f"     - Exact match: {context['exact_match'][:80]}...")
                
            except Exception as e:
                print(f"❌ Query failed: {str(e)}")

    def test_context_retrieval(self):
        """Step 5: Test detailed context retrieval"""
        print_section("STEP 5: CONTEXT RETRIEVAL TESTING")
        
        # Get a specific chunk and test context retrieval
        search_results = self.rag_pipeline.similarity_search_with_context("never trust, always verify", k=1)
        
        if search_results:
            result = search_results[0]
            print_subsection("Testing Context Retrieval")
            
            print(f"Target chunk: {result['chunk_content'][:100]}...")
            print(f"Document ID: {result['document_id']}")
            print(f"Position: {result['start_char']}-{result['end_char']}")
            
            # Test context retrieval with different window sizes
            for context_size in [100, 300, 500]:
                try:
                    context = self.rag_pipeline.get_document_context(
                        document_id=result['document_id'],
                        start_char=result['start_char'],
                        end_char=result['end_char'],
                        context_chars=context_size
                    )
                    
                    print(f"\nContext window ({context_size} chars):")
                    print(f"  Before: {context['context_before']}")
                    print(f"  Match: {context['exact_match']}")
                    print(f"  After: ...{context['context_after']}")
                    print(f"  Total context: {len(context['full_context'])} characters")
                    
                except Exception as e:
                    print(f"❌ Context retrieval failed: {str(e)}")

    def test_enhanced_search(self):
        """Step 6: Test enhanced search capabilities"""
        print_section("STEP 6: ENHANCED SEARCH TESTING")
        
        query = "security frameworks and best practices"
        print_subsection(f"Enhanced Search: '{query}'")
        
        # Test enhanced search with context
        results = self.rag_pipeline.enhanced_search(
            query=query,
            k=3,
            include_context=True,
            context_chars=250
        )
        
        print(f"Found {len(results)} enhanced results")
        
        for i, result in enumerate(results, 1):
            print(f"\nResult {i}:")
            print(f"  Score: {result['similarity_score']:.4f}")
            print(f"  File: {result['filename']}")
            print(f"  Chunk {result['chunk_index']}: {result['chunk_content'][:120]}...")
            
            if result.get('document_context'):
                print(f"  Context: {len(result['document_context']['full_context'])} chars available")

    def run_complete_test(self):
        """Run the complete enhanced pipeline test"""
        print_section("ENHANCED RAG PIPELINE - COMPLETE FLOW TEST")
        print("Testing: Document → Original Storage + Enhanced Chunks → Vector Store → Query → Context Retrieval → Enhanced Response")
        
        try:
            # Step 1: Create test documents
            self.create_test_documents()
            
            # Step 2: Upload and process documents
            self.upload_and_process_documents()
            
            # Step 3: Verify vector storage
            self.test_vector_storage_verification()
            
            # Step 4: Test enhanced queries
            self.test_enhanced_queries()
            
            # Step 5: Test context retrieval
            self.test_context_retrieval()
            
            # Step 6: Test enhanced search
            self.test_enhanced_search()
            
            print_section("TEST SUMMARY")
            print(f"✅ Documents processed: {len(self.processed_docs)}")
            print(f"✅ Vector storage verified")
            print(f"✅ Enhanced queries tested")
            print(f"✅ Context retrieval tested")
            print(f"✅ Enhanced search tested")
            print("\n🎉 ENHANCED RAG PIPELINE TEST COMPLETED SUCCESSFULLY!")
            
        except Exception as e:
            print(f"\n❌ TEST FAILED: {str(e)}")
            raise


if __name__ == "__main__":
    # Run the complete test suite
    tester = EnhancedRAGTester()
    tester.run_complete_test()