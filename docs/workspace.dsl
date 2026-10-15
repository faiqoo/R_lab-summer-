workspace "Islandora RAG Chatbot" "C4 model for the LLM-based chatbot extension to Islandora (Summer 2026 lab)." {

    !identifiers hierarchical

    model {
        researcher = person "Researcher / Archivist" "Browses the digital repository, uploads content, and asks the chatbot questions about the collection."

        islandora = softwareSystem "Islandora + RAG Extension" "Digital repository platform extended with an LLM chatbot for semantic Q&A over collection content." {

            # Groups mirror the subgraphs from the original flowchart.

            group "Main Application" {
                drupal = container "Drupal" "Repository front-end, content model, and event publishing. Hosts the chatbot module." "Drupal / PHP" {
                    chatUI = component "Chatbot UI" "Chat interface embedded in the repository UI; forwards questions to the RAG Chatbot Service." "Drupal module" {
                        tags "New"
                    }
                }
            }

            group "Storage" {
                existdb = container "eXistdb" "Stores TEI XML documents." "eXist-db" {
                    tags "Database"
                }
                mariadb = container "MariaDB" "Stores Drupal content metadata." "MariaDB" {
                    tags "Database"
                }
                fcrepo = container "Fedora (FCRepo)" "Stores binary files and original assets." "Fedora Commons" {
                    tags "Database"
                }
            }

            group "Message Broker" {
                activemq = container "ActiveMQ" "Message broker. Publishes repository events (e.g. media created) to consumers via STOMP." "Apache ActiveMQ" {
                    tags "Queue"
                }
            }

            group "Message Consumers" {
                nodered = container "Node-RED" "Event-driven orchestrator. Consumes repository events and forwards new documents for derivative generation and indexing." "Node-RED"
                alpaca = container "Alpaca" "Routes broker events to Crayfish microservices." "Java / Camel"
            }

            group "Processing Services" {
                bentoml = container "BentoML" "Generates TEI XML derivatives." "BentoML / Python"
                crayfish = container "Crayfish Services" "Generates OCR text, thumbnails, and audio derivatives." "PHP microservices"
            }

            group "RAG Extension" {
                embed = container "Embedding Service" "Turns text into dense embedding vectors on the GPU, for both document chunks and questions." "FastAPI / sentence-transformers (bge-m3)" {
                    tags "New"
                }
                vectordb = container "Vector Database" "Stores dense and sparse vectors with chunk metadata and runs hybrid search fused with RRF." "Qdrant" {
                    tags "Database" "New"
                }
                llm = container "LLM Service" "Generates answers and rewrites follow-up questions. Accessed through the HAWKI gateway provided by the VCRP." "HAWKI API" {
                    tags "New"
                }
                ragsvc = container "RAG Chatbot Service (ragsvc)" "Offers the query endpoints and the ingestion endpoints. Chunks and indexes documents, retrieves chunks for a question, builds the prompt and calls the LLM." "FastAPI / Python (LangGraph)" {
                    tags "New"
                    chatApi = component "Chat API" "Endpoints /chat and /chat/stream. Checks the API key and passes question and session ID to the query pipeline." "FastAPI router"
                    ingestApi = component "Ingestion API" "Endpoints /ingest, /ingest/async and /status/{job_id}. Stores uploads temporarily and tracks background jobs." "FastAPI router"
                    pipeline = component "Query Pipeline" "Keeps the session history, rewrites follow-up questions, applies the relevance guard, builds the prompt and generates the answer." "LangGraph"
                    searcher = component "Hybrid Searcher" "Runs the dense and BM25 search and returns the fused top-k chunks with their dense similarity." "Python, Qdrant client"
                    processor = component "Document Processor" "Extracts text from PDF and .txt files and splits it into structure-aware chunks." "Python, PyMuPDF"
                    encoders = component "Encoders" "Dense encoder (client of the embedding service) and local BM25 sparse encoder." "httpx, FastEmbed"
                    llmClient = component "LLM Client" "Sends prompts to the HAWKI API with retry and backoff and parses the responses." "httpx, tenacity"
                }
            }
        }

        # ── User interactions ──────────────────────────────────────
        researcher -> islandora.drupal "Browses & uploads content" "HTTPS"
        researcher -> islandora.drupal.chatUI "Asks questions, receives answers" "HTTPS"

        # ── Drupal → storage ───────────────────────────────────────
        islandora.drupal -> islandora.existdb "Stores TEI XML"
        islandora.drupal -> islandora.mariadb "Stores metadata"
        islandora.drupal -> islandora.fcrepo "Stores files / binaries"

        # ── Event messaging (async, point-to-point queues) ────────
        # A queue is a data store: producer adds messages, consumer takes them.
        # Tagged "Async" so messaging is drawn dashed, distinct from sync HTTP calls.
        islandora.drupal -> islandora.activemq "Publishes media-created events" "STOMP" "Async"
        islandora.activemq -> islandora.nodered "Delivers events" "STOMP" "Async"
        islandora.activemq -> islandora.alpaca "Delivers events" "STOMP" "Async"

        # ── Derivative generation ──────────────────────────────────
        islandora.nodered -> islandora.bentoml "Requests TEI XML derivative" "HTTP"
        islandora.bentoml -> islandora.drupal "Posts TEI XML derivative" "HTTP"
        islandora.alpaca -> islandora.crayfish "Requests OCR / thumbnails / audio" "HTTP"
        islandora.crayfish -> islandora.drupal "Posts derivatives (OCR text, thumbnails)" "HTTP"

        # ── RAG indexing path (Node-RED uploads the PDF; ragsvc does the
        # chunk/embed/upsert work, Node-RED never calls embed/vectordb
        # directly) ─────────────────────────────────────────────────
        islandora.nodered -> islandora.ragsvc.ingestApi "Uploads new documents for indexing" "HTTP POST /ingest"
        islandora.drupal -> islandora.ragsvc.ingestApi "Queue worker uploads new documents" "HTTP POST /ingest"
        islandora.ragsvc.ingestApi -> islandora.ragsvc.processor "Chunks uploaded documents"
        islandora.ragsvc.ingestApi -> islandora.ragsvc.encoders "Encodes chunks"
        islandora.ragsvc.encoders -> islandora.embed "Sends text, receives dense vectors" "HTTP / JSON"
        islandora.ragsvc.ingestApi -> islandora.vectordb "Upserts dense and sparse vectors with chunk metadata"

        # ── RAG query path (RAG Chatbot Service orchestrates) ──────
        islandora.drupal.chatUI -> islandora.ragsvc.chatApi "Forwards user question, receives answer" "HTTP POST /chat"
        islandora.ragsvc.chatApi -> islandora.ragsvc.pipeline "Runs the question"
        islandora.ragsvc.pipeline -> islandora.ragsvc.searcher "Retrieves chunks"
        islandora.ragsvc.searcher -> islandora.ragsvc.encoders "Encodes the question"
        islandora.ragsvc.pipeline -> islandora.ragsvc.llmClient "Rewrites questions, generates answers"
        islandora.ragsvc.searcher -> islandora.vectordb "Hybrid (dense and BM25) search for the top-k chunks"
        islandora.ragsvc.llmClient -> islandora.llm "Sends prompt, receives generated answer" "HTTPS"

        # ── Physical deployment: what runs where ───────────────────
        deploymentEnvironment "Development" {
            deploymentNode "Developer Laptop / isle-dc" "Docker Compose stack. Also runs dev tooling (search_repl, eval harness) that reaches the VM over the same SSH tunnel Node-RED uses." "macOS + Docker" {
                containerInstance islandora.drupal
                containerInstance islandora.activemq
                containerInstance islandora.nodered
                containerInstance islandora.alpaca
                containerInstance islandora.bentoml
                containerInstance islandora.crayfish
                containerInstance islandora.existdb
                containerInstance islandora.mariadb
                containerInstance islandora.fcrepo
            }

            deploymentNode "University GPU VM" "Reachable from the laptop only via SSH tunnel over the uni VPN, because the VM firewall allows SSH inbound only. Drupal cannot call ragsvc directly until university IT opens port 8000 for Drupal's actual origin." "Rootless podman + systemd --user" {
                deploymentNode "qdrant.service" "systemd-managed, bare process" {
                    containerInstance islandora.vectordb
                }
                deploymentNode "embedsvc.service" "systemd-managed, bare process, uses the VM's GPU" {
                    containerInstance islandora.embed
                }
                deploymentNode "ragsvc.service" "systemd-managed podman container, --network host" {
                    containerInstance islandora.ragsvc
                }
            }

            deploymentNode "HAWKI Gateway" "chat.edu-ki-rlp.de, provided by the VCRP, public HTTPS. Reached directly, no VPN or tunnel needed." "External" {
                containerInstance islandora.llm
            }
        }
    }

    views {
        systemContext islandora "SystemContext" "The repository system and who uses it." {
            include *
            autolayout lr
        }

        container islandora "Containers" "Existing Islandora services (blue) and the new RAG extension (green)." {
            include *
            autolayout lr
        }

        container islandora "RagContainers" "The RAG extension (green) and the existing containers it connects to (blue)." {
            include researcher islandora.drupal islandora.activemq islandora.nodered islandora.ragsvc islandora.embed islandora.vectordb islandora.llm
            autolayout tb
        }

        component islandora.ragsvc "RagsvcComponents" "Components inside the RAG service container." {
            include *
            autolayout tb
        }

        component islandora.drupal "DrupalComponents" "The new Chatbot UI component added to Drupal." {
            include islandora.drupal.chatUI
            autolayout lr
        }

        deployment islandora "Development" "Deployment" "What runs where, and how the pieces communicate across the laptop / GPU VM / HAWKI boundary." {
            include *
            autolayout lr
        }

        styles {
            element "Person" {
                shape person
                background #08427b
                color #ffffff
            }
            element "Software System" {
                background #1168bd
                color #ffffff
            }
            element "Container" {
                background #438dd5
                color #ffffff
            }
            element "Component" {
                background #85bbf0
                color #000000
            }
            element "Database" {
                shape cylinder
            }
            element "Queue" {
                shape pipe
            }
            element "New" {
                background #28a745
                color #ffffff
            }

            # Messaging (STOMP queue) relationships: dashed = async,
            # to distinguish them from synchronous HTTP calls (solid).
            relationship "Async" {
                dashed true
                color #6b6b6b
            }
        }
    }
}
