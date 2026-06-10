workspace "Islandora RAG Chatbot" "C4 model for the LLM-based chatbot extension to Islandora (Summer 2026 lab)." {

    !identifiers hierarchical

    model {
        researcher = person "Researcher / User" "Browses the digital repository, uploads content, and asks the chatbot questions about the collection."

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
                nodered = container "Node-RED" "Event-driven orchestrator. Consumes broker events and coordinates derivative + embedding generation." "Node-RED"
                alpaca = container "Alpaca" "Routes broker events to Crayfish microservices." "Java / Camel"
            }

            group "Processing Services" {
                bentoml = container "BentoML" "Generates TEI XML derivatives." "BentoML / Python"
                crayfish = container "Crayfish Services" "Generates OCR text, thumbnails, and audio derivatives." "PHP microservices"
            }

            group "RAG Extension" {
                embed = container "Embedding Service" "Stateless service: turns text into an embedding vector and returns it to the caller." "e.g. sentence-transformers" {
                    tags "New"
                }
                vectordb = container "Vector Database" "Stores embedding vectors and chunk metadata; serves similarity search." "ChromaDB" {
                    tags "Database" "New"
                }
                llm = container "LLM Service" "Generates answers from augmented prompts." "Llama via Ollama" {
                    tags "New"
                }
                ragsvc = container "RAG Chatbot Service" "Orchestrates the query path: embeds the question, retrieves chunks, builds the augmented prompt, and calls the LLM." "e.g. FastAPI / Python" {
                    tags "New"
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

        # ── RAG indexing path (Node-RED orchestrates the write) ────
        islandora.nodered -> islandora.embed "Sends extracted text, receives embedding vector" "HTTP / JSON"
        islandora.nodered -> islandora.vectordb "Stores embedding + chunk metadata"

        # ── RAG query path (RAG Chatbot Service orchestrates) ──────
        islandora.drupal.chatUI -> islandora.ragsvc "Forwards user question, receives answer" "HTTP / JSON"
        islandora.ragsvc -> islandora.embed "Sends query text, receives embedding vector" "HTTP / JSON"
        islandora.ragsvc -> islandora.vectordb "Similarity search for top-k chunks"
        islandora.ragsvc -> islandora.llm "Sends augmented prompt, receives generated answer" "HTTP"
    }

    views {
        systemContext islandora "SystemContext" "The repository system and who uses it." {
            include *
            autolayout lr
        }

        container islandora "Containers" "Existing Islandora services (blue) and the new RAG extension (green), grouped as in the original flowchart." {
            include *
            autolayout lr
        }

        component islandora.drupal "DrupalComponents" "The Chatbot UI as a component inside the Drupal container." {
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
