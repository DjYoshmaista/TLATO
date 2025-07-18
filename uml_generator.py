# uml_generator.py
# Requires the 'diagrams' library: pip install diagrams
# Requires Graphviz to be installed on your system.

from diagrams import Diagram, Cluster, Edge
from diagrams.programming.language import Python
from diagrams.onprem.compute import Server # Represents User/External API
from diagrams.generic.storage import Storage # Represents Filesystem I/O
from diagrams.generic.device import Mobile # Placeholder for User Interaction

# Define the diagram context
# direction="LR" (Left to Right) is often good for pipelines
with Diagram("TLATOv4.1 Execution Pipeline and Interactions", show=False, direction="LR", filename="tlato_v4_1_uml_corrected"):

    # --- External Entities ---
    user = Mobile("User")
    filesystem = Storage("Filesystem\n(Data, Repos, Logs, Models)")
    ollama_api = Server("Ollama API\n(Labeling, Synthetic Data)")

    # --- Main Application Flow ---
    with Cluster("User Interface & Orchestration"):
        main_py = Python("main.py")
        orchestrator = Python("SystemOrchestrator\n(in main.py)")
        m1_py = Python("m1.py\n(Data Submenu)")

    # --- Core Processing Modules (src) ---
    with Cluster("src/data - Data Handling"):
        constants_py = Python("constants.py")
        # processing_py = Python("processing.py") # The classes within are more specific
        repo = Python("DataRepository\n(in processing.py)")
        processor = Python("DataProcessor\n(in processing.py)")
        tokenizer_cls = Python("Tokenizer\n(in processing.py)")
        readers_py = Python("readers.py")
        # loaders_py = Python("loaders.py") # The class within is more specific
        dataloader = Python("EnhancedDataLoader\n(in loaders.py)")
        synthetic_py = Python("synthetic.py")
        synth_gen = Python("SyntheticDataGenerator\n(in synthetic.py)")

    with Cluster("src/core - Models & Core Logic"):
        models_py = Python("models.py")
        model = Python("ZoneClassifier\n(in models.py)")
        zones_py = Python("zones.py (Incomplete)")
        # Note: repo_handler.py and attention.py identified as likely redundant and omitted for clarity

    with Cluster("src/training - Model Training"):
        trainer_py = Python("trainer.py")
        trainer_cls = Python("EnhancedTrainer\n(in trainer.py)") # Renamed to avoid conflict with var name

    with Cluster("src/analysis - Analysis Tasks"):
        labeler_py = Python("labeler.py (Assumed)")
        labeler_cls = Python("SemanticLabeler\n(Assumed, via processor)") # Renamed

    with Cluster("src/utils - Utilities"):
        config_py = Python("config.py")
        logger_py = Python("logger.py")
        helpers_py = Python("helpers.py")
        gpu_switch_py = Python("gpu_switch.py")
        hashing_py = Python("hashing.py")
        compression_py = Python("compression.py")

    # --- Define Relationships and Flow ---

    # User starts the application
    user >> main_py
    main_py >> orchestrator

    # Orchestrator Setup and Main Menu Calls
    orchestrator >> Edge(label="Loads Config") >> config_py
    orchestrator >> Edge(label="Uses Logger") >> logger_py
    orchestrator >> Edge(label="Checks GPU") >> gpu_switch_py
    orchestrator >> Edge(label="Calls Data Menu") >> m1_py
    orchestrator >> Edge(label="Initiates Training (Menu)") >> trainer_cls
    orchestrator >> Edge(label="Loads Model (Menu)") >> model # Model instance
    model >> Edge(label="Uses load_state helper") >> helpers_py # Model loading uses helpers


    # m1.py (Data Submenu) Interactions
    m1_py >> Edge(label="Manages Repositories") >> repo
    m1_py >> Edge(label="Initiates Processing") >> processor
    m1_py >> Edge(label="Initiates Tokenization") >> tokenizer_cls
    m1_py >> Edge(label="Generates Synth Data (Menu)") >> synth_gen
    m1_py >> Edge(label="Uses Constants") >> constants_py
    m1_py >> Edge(label="Uses Config") >> config_py
    m1_py >> Edge(label="Uses Logger") >> logger_py
    m1_py >> Edge(label="Uses Utils") >> helpers_py
    m1_py >> Edge(label="Uses Hashing") >> hashing_py

    # DataRepository Interactions
    repo >> Edge(label="R/W Repo CSV/Index") >> filesystem
    repo >> Edge(label="Schema/Paths") >> constants_py
    repo >> Edge(label="File Hashes") >> hashing_py
    repo >> Edge(label="Zstandard R/W") >> compression_py
    repo >> Edge(label="Logs Actions") >> logger_py
    repo >> Edge(label="Backend (pd/cudf)") >> config_py

    # DataProcessor Interactions
    processor >> Edge(label="Gets/Updates File Status") >> repo
    processor >> Edge(label="Reads Files") >> readers_py
    processor >> Edge(label="Calls Labeling (Assumed)") >> labeler_cls
    processor >> Edge(label="Saves Processed Data") >> filesystem
    processor >> Edge(label="Paths/Status") >> constants_py
    processor >> Edge(label="Logs Actions") >> logger_py
    processor >> Edge(label="Processing Params") >> config_py

    # Tokenizer Interactions
    tokenizer_cls >> Edge(label="Gets Files, Updates Status") >> repo
    tokenizer_cls >> Edge(label="Saves Tokens (.pt)") >> filesystem
    tokenizer_cls >> Edge(label="Paths/Status") >> constants_py
    tokenizer_cls >> Edge(label="Logs Actions") >> logger_py
    tokenizer_cls >> Edge(label="Model Name") >> config_py

    # Readers Module Interactions
    readers_py >> Edge(label="Reads Source Files") >> filesystem
    readers_py >> Edge(label="File Types") >> constants_py
    readers_py >> Edge(label="Logs Read Ops") >> logger_py

    # Synthetic Data Generator Interactions
    synth_gen >> Edge(label="Generates Samples") >> ollama_api
    synth_gen >> Edge(label="Saves Samples (JSONL)") >> filesystem
    synth_gen >> Edge(label="API Endpoint, Model") >> config_py
    synth_gen >> Edge(label="Logs Generation") >> logger_py

    # Trainer Interactions
    trainer_cls >> Edge(label="Gets Batches") >> dataloader
    trainer_cls >> Edge(label="Trains") >> model # Operates on the model instance
    trainer_cls >> Edge(label="Save/Load Checkpoints") >> helpers_py
    trainer_cls >> Edge(label="R/W Checkpoints/Metrics") >> filesystem
    trainer_cls >> Edge(label="Logs Training Progress") >> logger_py
    trainer_cls >> Edge(label="Training Params") >> config_py
    trainer_cls >> Edge(label="Sets Device") >> gpu_switch_py

    # DataLoader Interactions
    dataloader >> Edge(label="Loads Tokens (.pt)") >> filesystem
    dataloader >> Edge(label="Gets File List (Potentially)") >> repo
    dataloader >> Edge(label="Paths") >> constants_py
    dataloader >> Edge(label="Logs Loading") >> logger_py
    dataloader >> Edge(label="Zstandard Read") >> compression_py # If tokens are compressed

    # Semantic Labeler Interactions (Assumed)
    labeler_cls >> Edge(label="Sends Text for Labeling") >> ollama_api
    labeler_cls >> Edge(label="Logs Labeling") >> logger_py

    # --- General Utility Dependencies (simplified, as many modules use them) ---
    # Explicit connections are preferred, but for brevity, these show general use.
    # Add more specific Edge connections if a utility is critical to a specific interaction.
    _utility_users = [
        orchestrator, m1_py, repo, processor, tokenizer_cls, readers_py,
        synth_gen, trainer_cls, dataloader, model, labeler_cls, helpers_py, gpu_switch_py
    ]

    for user_module in _utility_users:
        # Common utilities, these edges can make the diagram very dense
        # Only add if crucial or keep general and describe in text.
        # Example: If every module logs:
        # user_module >> Edge(style="dotted", color="grey") >> logger_py
        # user_module >> Edge(style="dotted", color="grey") >> config_py
        # user_module >> Edge(style="dotted", color="grey") >> constants_py
        pass # Keeping it less cluttered; specific utility uses are already drawn above.

    # Specific utils that might be broadly used and not yet explicitly linked from all users
    helpers_py >> Edge(label="Uses Hashing") >> hashing_py # if helpers internally use hashing
    helpers_py >> Edge(label="Uses Compression") >> compression_py # if helpers internally use compression
    helpers_py >> Edge(label="Accesses Filesystem") >> filesystem # For save_state etc.