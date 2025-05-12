# uml_generator.py
# Requires the 'diagrams' library: pip install diagrams

from diagrams import Diagram, Cluster, Edge
from diagrams.programming.language import Python
from diagrams.onprem.compute import Server # Represents User/External API
from diagrams.generic.storage import Storage # Represents Filesystem I/O
from diagrams.generic.device import Mobile # Placeholder for User Interaction

# Define the diagram context
# direction="LR" (Left to Right) is often good for pipelines
with Diagram("TLATOv4.1 Execution Pipeline and Interactions", show=False, direction="LR", filename="tlato_v4_1_uml"):

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
        processing_py = Python("processing.py")
        repo = Python("DataRepository\n(in processing.py)")
        processor = Python("DataProcessor\n(in processing.py)")
        tokenizer_cls = Python("Tokenizer\n(in processing.py)") # Renamed to avoid conflict
        readers_py = Python("readers.py")
        loaders_py = Python("loaders.py")
        dataloader = Python("EnhancedDataLoader\n(in loaders.py)")
        synthetic_py = Python("synthetic.py")
        synth_gen = Python("SyntheticDataGenerator\n(in synthetic.py)")

    with Cluster("src/core - Models & Core Logic"):
        # Note: repo_handler.py and attention.py identified as likely redundant
        models_py = Python("models.py")
        model = Python("ZoneClassifier\n(in models.py)")
        zones_py = Python("zones.py (Incomplete)")

    with Cluster("src/training - Model Training"):
        trainer_py = Python("trainer.py")
        trainer = Python("EnhancedTrainer\n(in trainer.py)")

    with Cluster("src/analysis - Analysis Tasks"):
        # Note: labeler.py content was incorrect in snippets
        labeler_py = Python("labeler.py (Assumed)")
        labeler = Python("SemanticLabeler\n(Assumed, via processor)")

    with Cluster("src/utils - Utilities"):
        config_py = Python("config.py")
        logger_py = Python("logger.py")
        helpers_py = Python("helpers.py")
        gpu_switch_py = Python("gpu_switch.py")
        hashing_py = Python("hashing.py")
        compression_py = Python("compression.py")
        # file_processor.py seems unused based on imports

    # --- Define Relationships and Flow ---

    # User starts the application
    user >> main_py >> orchestrator

    # Orchestrator Setup and Main Menu Calls
    orchestrator >> config_py : "Loads Config"
    orchestrator >> logger_py : "Uses Logger"
    orchestrator >> gpu_switch_py : "Checks GPU"
    orchestrator >> m1_py : "Calls Data Menu"
    # Orchestrator also likely calls Trainer (via menu option)
    orchestrator >> trainer : "Initiates Training (Menu)"
    orchestrator >> model : "Loads Model (Menu)"
    orchestrator >> helpers_py : "Uses load_state"

    # m1.py (Data Submenu) Interactions
    m1_py >> repo : "Manages Repositories"
    m1_py >> processor : "Initiates Processing"
    m1_py >> tokenizer_cls : "Initiates Tokenization"
    m1_py >> synth_gen : "Generates Synth Data (Menu)" # Assuming menu option exists
    m1_py >> constants_py : "Uses Constants"
    m1_py >> config_py : "Uses Config"
    m1_py >> logger_py : "Uses Logger"
    m1_py >> helpers_py : "Uses Utils"
    m1_py >> hashing_py : "Uses Hashing"

    # DataRepository Interactions
    repo >> filesystem : "R/W Repo CSV/Index"
    repo >> constants_py : "Schema/Paths"
    repo >> hashing_py : "File Hashes"
    repo >> compression_py : "Zstandard R/W"
    repo >> logger_py : "Logs Actions"
    repo >> config_py : "Backend (pd/cudf)"
    # Connection to helpers.py _scan etc. might exist if not fully refactored

    # DataProcessor Interactions
    processor >> repo : "Gets/Updates File Status"
    processor >> readers_py : "Reads Files"
    processor >> labeler : "Calls Labeling (Assumed)"
    processor >> filesystem : "Saves Processed Data"
    processor >> constants_py : "Paths/Status"
    processor >> logger_py : "Logs Actions"
    processor >> config_py : "Processing Params"

    # Tokenizer Interactions
    tokenizer_cls >> repo : "Gets Files, Updates Status"
    tokenizer_cls >> filesystem : "Saves Tokens (.pt)"
    tokenizer_cls >> constants_py : "Paths/Status"
    tokenizer_cls >> logger_py : "Logs Actions"
    tokenizer_cls >> config_py : "Model Name"

    # Readers Module Interactions
    readers_py >> filesystem : "Reads Source Files"
    readers_py >> constants_py : "File Types"
    readers_py >> logger_py : "Logs Read Ops"

    # Synthetic Data Generator Interactions
    synth_gen >> ollama_api : "Generates Samples"
    synth_gen >> filesystem : "Saves Samples (JSONL)"
    synth_gen >> config_py : "API Endpoint, Model"
    synth_gen >> logger_py : "Logs Generation"

    # Trainer Interactions
    trainer >> dataloader : "Gets Batches"
    trainer >> model : "Forward/Backward Pass"
    trainer >> helpers_py : "Save/Load Checkpoints"
    trainer >> filesystem : "R/W Checkpoints/Metrics"
    trainer >> logger_py : "Logs Training Progress"
    trainer >> config_py : "Training Params"
    trainer >> gpu_switch_py : "Sets Device"

    # DataLoader Interactions
    dataloader >> filesystem : "Loads Tokens (.pt)"
    dataloader >> repo : "Gets File List (Potentially)"
    dataloader >> constants_py : "Paths"
    dataloader >> logger_py : "Logs Loading"
    dataloader >> compression_py : "Zstandard Read" # If tokens are compressed

    # Model Interactions
    model >> helpers_py : "Loaded/Saved via helpers" # Indirectly via orchestrator/trainer

    # Semantic Labeler Interactions (Assumed)
    labeler >> ollama_api : "Sends Text for Labeling"
    labeler >> logger_py : "Logs Labeling"

    # --- General Utility Dependencies ---
    # Most modules depend on logger and config
    _utils_users = [main_py, m1_py, repo, processor, tokenizer_cls, readers_py, synth_gen, trainer, dataloader, model, labeler, helpers_py, gpu_switch_py]
    for user_module in _utils_users:
        user_module >> logger_py : "Imports"
        user_module >> config_py : "Imports"
        user_module >> constants_py: "Imports" # Many use constants too

    # Specific utils
    helpers_py >> hashing_py
    helpers_py >> compression_py
    helpers_py >> filesystem # Save state, etc.