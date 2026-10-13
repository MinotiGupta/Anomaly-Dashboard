# Project Report: Sensor Anomaly Detection and Measurement Integrity

**Project snapshot:** 28 September 2026  
**Repository:** `C:\IITM_project`  
**Scope:** Existing sensor-data analysis/dashboard plus the measurement-integrity, edge-processing, physics-detection, and thermal-bias work present in the repository at the snapshot date.

## 1. Executive summary

This repository contains a sensor time-series anomaly-detection project with two related but distinct systems:

1. **The established CSV analysis dashboard** accepts instrument log files, extracts sensor channels, trains or loads a PyTorch LSTM autoencoder, scores sequences using reconstruction error, presents interactive charts, and accepts researcher feedback for later retraining.
2. **A newer measurement-integrity backend** defines a versioned per-sample contract and implements edge-side hardware checks, timing checks, physics-based checks, quality classes, thermal-bias annotations, persistent detector state, a local outbox, dashboard-side storage, configuration routes, and diagnostic/analytics API routes.

The newer backend is designed to retain each raw measurement and add flags or annotations. A science view excludes only `implausible` records while retaining the raw record. The parameter-fitting method now uses that science view and reports how many records it excluded.

The work is **not yet a complete deployed hardware-to-dashboard system**. The repository does not include sensor-driver acquisition code or an outbox uploader. The React interface currently calls the CSV workflow and does not call the new measurement, diagnostics, or physical-configuration routes. The edge processor can be initialized from saved configurations through a backend factory, but that factory is not an acquisition service or a network synchronizer. CSV source values are now preserved during loading and annotated, but malformed/missing values still need model-path handling that avoids imputation.

## 2. Project purpose and measurement problem

The project aims to identify unusual or faulty behavior in multi-channel temperature and process sensor data. The original dashboard uses learned time-series behavior to find patterns the model has not seen. The newer work addresses a separate measurement-science concern: a sensor can report a value that looks statistically unusual because of a hardware fault, physical impossibility, sensor degradation, or an expected measurement bias.

The included reference document, `Ai4Energy_Anomaly_Detection_Approach (1).pdf`, informs the newer approach. The requirements captured in `dashboard/backend/MEASUREMENT_CONTRACT.md` prioritize signal integrity, physical plausibility, measured thermal response, explicit quality classes, and explanations of expected thermal measurement effects.

The design principle is that an anomaly detector must not silently rewrite its evidence. Raw samples are retained; detector results are additive. A hard hardware or physics violation may make a sample ineligible for a derived science dataset, but the original record remains queryable. Statistical surprise alone is not a reason to suppress a measurement.

## 3. Data in the repository

### 3.1 Data families

The `data/` directory contains instrument CSVs organized into these groups:

| Directory | What is present | Evidence visible in file headers |
|---|---|---|
| `data/Biomass/` | Gasifier/biomass runs, including cleaned and uncleaned files and run names referring to different flow conditions | Agilent/Keysight 34970A metadata in a sampled file |
| `data/cataluminiscence/` | Cataluminescence runs, including 2025-11-21 runs and a cleaned export | DAQ970A metadata in a sampled file |
| `data/Swiss Roll/` | Two low-flow runs with `2slpm` in their filenames | 34970A metadata in a sampled file |
| `data/default/` | A gasifier run | A 34970A-style data family by project naming and classification rules |
| `data/uploaded/` | Copies of several files also represented in the named experiment folders | Uploaded working copies |

The CSVs contain instrument metadata such as instrument address, model, serial number, firmware, start/stop time, and channel configuration. Data rows are instrument scans with timestamps and one or more measured channels. The exact scientific meaning, units, calibration status, normal/fault labels, and channel-to-probe mapping are not uniformly encoded in the repository. Those need to be confirmed from experiment records or the instrument logs before making scientific claims from model outputs.

Some filenames include `cleaned`; those files are already processed datasets. The repository does not contain a single data dictionary documenting every column or documenting the cleaning steps used to create each cleaned copy.

### 3.2 Research notebooks and trained artifacts

The root contains CSV-analysis and LSTM/KDE notebooks for biomass, cataluminescence, default, and Swiss Roll data, plus a `Complete_pipeline.ipynb` notebook. These notebooks preserve exploratory and experiment-specific analysis alongside the unified dashboard implementation.

`Saved_models/` contains PyTorch weight files for `swiss_roll`, `default`, `cataluminiscence`, and `biomass`. The dashboard also has code to save a model's weights, fitted scaler, and JSON metadata after training. Model outputs should be treated as data-driven anomaly scores, not as proof of a physical sensor fault.

## 4. Existing CSV dashboard and model pipeline

### 4.1 Backend pipeline

`dashboard/backend/app.py` contains the Flask server and the primary legacy CSV pipeline:

1. A CSV is uploaded and stored by the upload endpoint.
2. The loader searches for a data header, reads the table, and identifies likely time and sensor columns.
3. Instrument/model and channel naming rules classify the data as biomass, cataluminescence, Swiss Roll, default, or unknown.
4. Instrument metadata is extracted from recognized Agilent/Keysight-style headers.
5. A dynamic LSTM autoencoder is trained or a saved model/scaler is loaded.
6. A sliding window is reconstructed; mean absolute reconstruction error (MAE) is used as the sequence score.
7. A KDE valley threshold is used where available, with a percentile fallback. A saved threshold is reused for stable inference when model metadata provides one.
8. Results include per-channel series and anomaly rows, error/confidence diagnostics, feature contributions, training loss, and channel statistics.

The hidden layer size is selected from the channel count, bounded between 16 and 64 units. The default sequence length is 10 samples. Training uses an L1 reconstruction loss and Adam optimization. The pipeline can train a new model, reuse saved artifacts, or retrain.

### 4.2 Human feedback and persistence

The dashboard provides a feedback flow in which a researcher marks candidate anomalies as true anomalies or false positives. The backend stores feedback in per-data-type SQLite databases. Feedback-aware retraining can increase the training weight of samples marked normal. Feedback is attached to timestamps and includes sensor values and model score/context.

Model artifacts are keyed by data family and a hash of the sorted sensor-column names so models trained for different channel sets are kept separate. Metadata includes the feature list, threshold/method, training time, epoch count, final loss, and whether feedback was used.

### 4.3 Frontend

The React/Vite frontend currently implements the CSV workflow. The main page supports file upload, run, rerun with feedback, reset, and result display. Result components cover time-series plots, anomaly tables, confidence and reconstruction-error distributions, training loss, feature contributions, and summary statistics.

The frontend API client currently calls `/api/upload`, `/api/run`, and feedback endpoints. It does not yet call the newer `/api/measurements`, `/api/diagnostics`, edge configuration, or physical-configuration endpoints. Therefore the new measurement-quality diagnostics are not yet part of the UI flow.

### 4.4 Model used: LSTM autoencoder, not a CNN-LSTM

The model used by the dashboard is a **multivariate LSTM autoencoder**. It is not a CNN-LSTM in the current implementation: the dashboard class `DynamicLSTMAutoencoder` contains LSTM layers and a linear output layer, and the experiment-specific notebooks inspected in this repository define LSTM autoencoders. No `Conv1d`, `Conv2d`, or convolutional/recurrent hybrid appears in those model definitions. Some notebooks contain alternative LSTM variants, but the dashboard's deployed pipeline uses the dynamic LSTM autoencoder described here. The phrase “CNN/LSTM” should therefore be corrected to “LSTM autoencoder” when describing the model currently wired into the application.

#### Why use an LSTM for these data?

The CSV observations are ordered in time. A reading is meaningful partly because of how it follows prior readings: temperature ramps, stable operating periods, process transitions, oscillations, sensor lag, and coordinated changes across channels. A model that treats every row as an unrelated sample would lose this temporal order.

An LSTM (Long Short-Term Memory recurrent network) processes a sequence one timestep at a time and carries a learned hidden state forward. Its gated state is designed to retain useful context across multiple timesteps while reducing the tendency of a basic recurrent network to lose older information during training. In this project, each timestep contains all selected sensor channels, so the network can learn temporal patterns within a channel and relationships represented jointly across the channels.

The autoencoder form is useful where the CSV runs do not provide a complete, trusted label for every normal and faulty sample. The model is trained to reconstruct observed windows. Windows whose reconstruction error is high are candidates for review because their joint time/channel pattern was not reconstructed as well as the patterns represented by the trained model. This is an unsupervised or weakly supervised anomaly-scoring approach; it does not establish the physical cause of an anomaly.

#### Dashboard architecture

For `F` sensor channels and a window of `T` timesteps, the model input is shaped as:

```text
batch_size x T x F
```

The dashboard pipeline uses `T = 10`. Before model use, the sensor columns are transformed using `StandardScaler`, so each channel is represented relative to the training data's fitted mean and standard deviation. The saved scaler is reused with its model so inference uses the same feature scaling as training.

The encoder LSTM reads the complete window and its final hidden state acts as a compressed representation of the window. The decoder repeats that encoded state for each of the `T` positions, runs a decoder LSTM, and uses a linear layer to reconstruct the `F` channel values at every timestep. The implementation uses `batch_first=True`; its model input and output are both batch-by-time-by-feature tensors.

The dashboard chooses the hidden size from the channel count:

```text
hidden_size = min(64, max(16, 4 x number_of_channels))
```

This gives the network a larger representation for datasets with more channels while imposing a lower and upper bound. For the saved biomass artifact with five channels, metadata records a hidden size of 20, consistent with this rule. The notebook experiments commonly use a fixed hidden size such as 64; the unified dashboard implementation is dynamic.

#### Training objective and anomaly score

The standard training objective is mean absolute reconstruction error (`L1Loss`) across the tensor elements. Training uses Adam, a default of 30 epochs, and batches of 16 sequences. The feedback-aware training variant applies a per-sequence weight: timestamps confirmed as normal receive weight 2, while other sequences receive weight 1. This encourages the model to reconstruct confirmed-normal examples accurately. The feedback path does not turn the task into a fully supervised classifier.

At scoring time, the backend calculates a mean absolute error for each complete window across time and channels:

```text
window_MAE = mean(abs(reconstructed_window - input_window))
```

A larger MAE indicates poorer reconstruction and is treated as greater anomaly evidence. The backend also computes per-feature reconstruction error for visualization. The anomaly table uses each sequence's final timestamp and corresponding original sensor values, while the time-series view overlays detected candidate points.

#### Threshold selection and persistence

The backend fits a Gaussian kernel density estimate (KDE) to reconstruction errors. If a local minimum is found in the estimated density, the first such valley is used as the decision threshold. If no local minimum is found, the implementation falls back to the 97th percentile. A model's threshold and the method used are stored in its metadata and reused for subsequent inference when available. The KDE curve is still generated for the dashboard's diagnostic view.

The model key depends on the data family and a hash of the sorted channel names, so models with different feature sets are separated. Saved artifacts include the PyTorch weights (`.pth`), fitted scaler (`_scaler.pkl`), and metadata (`_meta.json`). The metadata includes channel names, feature count, hidden size, training timestamp, number of epochs, final loss, feedback usage, threshold, and threshold method.

For example, the repository's `biomass_5ch_33eb7d11_meta.json` describes a five-channel biomass model trained for 30 epochs with final training loss 0.1613508672, no feedback used, and a stored KDE-valley threshold of 0.3263053283. Those values describe that artifact's training run; they are not performance metrics such as precision, recall, or accuracy, and the threshold is tied to the stored model/scaler and their training distribution.

#### Why not a CNN-LSTM here?

A one-dimensional CNN can be useful for detecting local motifs and reducing sequence length before a recurrent stage. A CNN-LSTM hybrid would add convolutional layers and a different set of kernel, padding, pooling, and receptive-field choices. The dashboard code does not contain those layers, and the project artifacts do not establish that a CNN-LSTM was trained for the current UI pipeline. We should not claim that architecture was used without a corresponding implementation and saved artifact.

The current LSTM autoencoder is a reasonable baseline for multichannel sequence reconstruction: it is comparatively direct to inspect, works on time-ordered windows, and does not require point-by-point anomaly labels. A CNN-LSTM could be evaluated as a future model comparison if there are enough runs and a defined evaluation protocol, but it should not be substituted into this report as the present model.

#### Interpretation and limitations of the model

- A high reconstruction error means “unlike the patterns reconstructed by this model,” not “sensor hardware is definitely faulty.” Physical constraints and signal-integrity checks provide a separate evidence path.
- The detector depends on representative training data. If training data contain faults, unusual operating regimes, or multiple regimes with very different distributions, the learned reconstruction behavior and threshold can be misleading.
- A threshold selected from the same run's KDE is distribution-dependent. Reusing a saved threshold improves consistency with the saved model, but does not prove the threshold generalizes to a different device, installation, sensor, or operating condition.
- The 97th-percentile fallback is a heuristic. It does not guarantee a 3% false-positive rate on future runs.
- The reported confidence score is a linear rescaling of an MAE above threshold relative to the largest MAE in the scored run. It is a display score, not a calibrated probability that a sample is faulty.
- Sequence-level scoring spreads evidence over a window and reports the final timestamp for that window. It is not a precise physical fault onset estimator.
- The current pipeline does not show a held-out labeled benchmark with accuracy, precision, recall, F1, false-alarm rate, or detection delay. The saved final training loss alone cannot validate anomaly-detection quality.
- Channel scaling, model keys, saved thresholds, and feature order must remain aligned. Changing channel selection or preprocessing can invalidate comparability with an existing artifact.
- The current legacy pipeline's handling of missing/nonnumeric raw sensor cells remains incomplete, as described in the limitations section below.

The newer edge physics checks complement this model rather than replace it. Statistical reconstruction error is treated as `unusual` under the quality policy; only hard hardware or physics constraints can create `implausible` flags and exclude records from the science view.

## 5. Versioned measurement contract

`dashboard/backend/measurement_contract.py` defines schema version `1.0` and the immutable `MeasurementRecord` model. It represents one sample with:

- Raw value and timezone-aware wall-clock timestamp.
- Monotonic uptime and boot ID.
- Device and channel IDs.
- Sensor and front-end enums.
- Sample interval.
- Optional raw hardware word and register payload.
- Detector ruleset and parameter-set versions.
- Process/event metadata, timing metadata, and optional oversample statistics.
- Quality flags and explanatory diagnostic annotations.

The model validates key fields and timezone awareness. `raw_payload()` serializes the complete record. Helper methods return copies with additional flags or diagnostics, leaving the original raw value untouched. `QualityFlag` requires a code, one of `implausible`, `suspicious`, or `unusual`, a message, detector identity, and timezone-aware detection time. `DiagnosticAnnotation` is a separate explanatory type and is not a quality flag.

This contract is for hardware-aware records. The legacy CSV path does not have boot IDs, monotonic time, raw registers, detector versions per sample, or all other provenance. Those fields must not be invented from ordinary CSV data.

## 6. Edge layer and signal-integrity checks

### 6.1 Offline edge processor

`dashboard/backend/edge_layer.py` provides `EdgeProcessor` and `EdgeStateStore`.

- `EdgeProcessor.process(record)` applies timing checks, then front-end-specific integrity checks, then physics and thermal-bias analysis if a physical configuration is loaded.
- Quality flags are deduplicated by code and passed through the quality authority policy.
- Detector state is updated and stored per device/channel in SQLite.
- The resulting record is appended to a JSON Lines outbox for later delivery.
- `summarize_interval()` computes sample count, mean, minimum, maximum, and sample standard deviation from oversamples; it does not overwrite those inputs.

The persistent state tracks measurement counts, boot observations, timing information, detector state, and DS18B20 CRC/ROM consistency statistics. The MAX31855 interface is recorded as lacking CRC and parity because those error-detection mechanisms are not provided by that device.

### 6.2 Timing checks

The edge processor can flag a boot ID change, monotonic-time regression, sample interval mismatch, or declared-versus-observed DS18B20 conversion delay mismatch. Conversion start/completion and cross-sensor offsets are carried in `TimingMetadata`. Rate calculations use monotonic time; wall-clock time remains for display/storage.

### 6.3 MAX31855

The processor contains checks for:

- All-zero and all-one 32-bit words.
- Reserved bits.
- Optional duplicate-read consistency.
- Summary fault-bit consistency against individual open/short indications.
- The lack of CRC/parity, recorded in persistent edge capability state.

These checks depend on the acquisition adapter providing the raw word and optional register fields. No MAX31855 device driver is present in this repository.

### 6.4 MAX31856

Separate checks are implemented for open thermocouple, over-voltage, under-voltage, thermocouple and cold-junction limit flags, configuration reset/readback mismatch, and configured thermocouple/cold-junction temperature bounds. The acquisition adapter must pass the expected and read-back configuration and individual fault-register conditions.

### 6.5 DS18B20

The processor supports scratchpad CRC-8 validation, raw-word all-zero/all-one checks, bus presence, ROM-ID matching, optional expected ROM ID, periodic bus ROM manifest verification, bus topology-change detection, and the exact 85 C power-on value condition when conversion has not completed. CRC failure count/rate and ROM/topology state are persisted per channel.

Checks that depend on hardware data are only active when that data is supplied. The repo does not include a one-wire bus acquisition implementation or a configured production ROM manifest.

## 7. Per-channel physical configuration

`dashboard/backend/physical_configuration.py` defines `ChannelPhysicalConfiguration` and `SensorPair`. A config includes device/board/node/channel identity; sensor/front-end; thermocouple junction style; probe diameter/location/insertion depth; measured thermal time constant and uncertainty; physical/source/sink bounds; noise floor; sample interval; datasheet resolution; tolerance class; related sensor pairs; and extensible metadata.

The configuration validator requires positive tau, a tau uncertainty smaller than tau, consistent temperature bounds, and a tau measurement method indicating commissioning-step measurement, online transient fit, or both. Thermocouple front-ends require a tolerance class. This prevents a catalogue tau value from being accepted as if it were measured.

The dashboard stores versioned per-channel configs in SQLite and provides PUT/GET routes. The recent addition in `app.py` provides `create_edge_processor(state_store, device_id, ...)`, which loads the device's latest stored per-channel configurations and supplies them to a new `EdgeProcessor`. The processor keeps that snapshot in memory for offline use; recreate it or call `set_physical_configurations()` after config changes.

This is a local backend configuration-loading path. It is not yet an edge-to-cloud configuration synchronization client, nor is there a deployment/runtime that starts the edge processor automatically.

## 8. Physics-based detection

`dashboard/backend/physics_detector.py` applies configuration-driven checks after signal-integrity checks:

1. **Thermodynamic bounds:** compares a sample against the intersection of configured physical and source/sink bounds.
2. **Slope limits:** estimates the sample-to-sample rate from monotonic time and compares with a first-order thermal response limit using measured tau and its uncertainty.
3. **Cold-soak self-check:** when event metadata marks an active cold soak and supplies a reference temperature, checks the channel against noise/resolution-based tolerance.
4. **High-pass energy:** computes a short-window residual RMS or consumes supplied high-pass RMS and compares it with configured noise/resolution.
5. **Two-sensor timing/residual:** compares explicit peer timing/value metadata or a passed peer record against expected offset and configured scale.
6. **Online tau/drift:** estimates a first-order tau from recent values and flags a mismatch from the configured measured tau.
7. **Gated drift:** uses CUSUM-like accumulation only during event metadata marked `quiescent_window`; an implausible flag prevents this update from triggering a drift result.

The first-order physical model guiding the work is `dT/dt = (T_infinity - T)/tau`. Hard physical violations are `implausible`; degradation signatures are `suspicious`; physically possible surprises are `unusual`.

Some checks require explicit event metadata, peer data, or high-pass inputs that an acquisition system must supply. They cannot be expected to fire from a temperature number alone.

## 9. Quality policy and science view

`dashboard/backend/quality_policy.py` sets the authority boundary:

- **Implausible:** hard hardware/physics constraints; excluded from the science view.
- **Suspicious:** a possible degradation/maintenance condition; retained in science data and surfaced diagnostically.
- **Unusual:** statistically surprising but physically possible; retained and surfaced.

Known statistical detectors, including KDE, reconstruction error, autoencoder, and statistical baseline signals, are normalized to `unusual` if they attempt to claim a stronger class. This prevents statistical rarity by itself from vetoing a sample.

`science_view()` filters only records with an `implausible` flag. It does not delete or alter the stored record. `DashboardStore` has raw and science list methods, diagnostics aggregation, and durable SQLite measurement storage keyed by measurement ID.

The recent parameter-fit change makes `fit_channel_parameters()` use only science-eligible records and expose `raw_record_count` and `excluded_implausible_count`. If every record in a channel is excluded, the sample count is zero and estimates are `None` rather than fabricated. This fit reports mean, a difference-based noise estimate, and a first-order tau estimate where enough data exist. These are exploratory estimates and require suitable excitation and sampling before being interpreted as calibrated parameters.

## 10. Thermal-bias diagnostics

`dashboard/backend/thermal_bias.py` emits `DiagnosticAnnotation` entries separately from quality flags. When the necessary context exists, it can annotate:

- Radiation exchange based on probe/wall temperatures, emissivity, and heat-transfer coefficient.
- Stem conduction based on mount temperature, insertion-depth/diameter ratio, and sample temperature.
- Recovery heating at high gas velocity.
- Thermocouple inhomogeneity when repositioning or a known signature is supplied.
- Cold-junction compensation drift after board-temperature movement.
- MAX31855 linearization difference when a reference difference is supplied.
- Junction-style effects from channel configuration.

These annotations explain measurement effects. They do not change the sample, create anomaly flags, or exclude data from the science view. Some estimates are simplified engineering diagnostics and need experimentally validated inputs; they should not be treated as a universal correction model.

## 11. Cloud/dashboard measurement API and storage

The Flask backend includes routes for:

| Route | Purpose |
|---|---|
| `POST /api/measurements` | Validate and store a batch of versioned measurement records; duplicate measurement IDs are ignored |
| `GET /api/measurements/<device_id>` | Retrieve raw records, optionally filtered by channel |
| `GET /api/measurements/<device_id>/science` | Retrieve the derived science view |
| `GET /api/diagnostics/<device_id>` | Aggregate record counts, quality classes, flags, and thermal annotations |
| `GET /api/analytics/<device_id>/parameters` | Return fitted per-channel parameters and exclusion counts |
| `PUT/GET /api/edge-config/<device_id>` | Save/retrieve a versioned edge configuration bundle |
| `PUT/GET /api/devices/<device_id>/channels/<channel_id>/physical-config` | Publish/retrieve a validated per-channel physical configuration |
| `GET /api/devices/<device_id>/physical-config` | List latest channel configurations for a device |

`DashboardStore` uses SQLite tables for measurements, edge configuration bundles, and per-channel physical configurations. Measurement payload JSON retains the complete serialized record. Its list API currently caps retrieval at 10,000 records per call by defaulting/capping the query limit.

## 12. Tests and verification assets

The backend includes pytest modules for:

- Measurement contract validation and serialization.
- Timing behavior.
- MAX31855/MAX31856/DS18B20 integrity checks.
- Quality-class authority and science-view filtering.
- Physics detector behavior.
- Physical configuration validation and API publishing.
- Measurement API and dashboard storage.
- Layer state persistence and edge output.
- Thermal-bias annotation behavior.

The tests are useful verification assets, but the current workspace environment did not have a Python interpreter/launcher available during the recent checks. Tests and module compilation therefore could not be executed here. `git diff --check` was run after the recent source edits and reported no whitespace errors.

## 13. Recent repository changes reflected in this report

The following changes were made during the current work session:

1. **Physical config loading:** Added a backend factory to initialize an `EdgeProcessor` with the latest saved per-device channel configurations. Added configuration-map validation and made channel config listing select the latest version per channel. Added a test for the loading path.
2. **CSV raw-value preservation:** Removed overload replacement and forward/back filling from the CSV loader. It now retains the sensor cells as loaded and adds row/channel quality annotations to the loader summary for missing, nonnumeric, non-finite, and overload values.
3. **Science-only parameter fitting:** Updated cloud-side parameter estimates to exclude implausible records while reporting raw and excluded counts. Added tests for partial and total exclusion cases.

## 14. Current limitations and remaining integration work

These are the main caveats visible from the repository:

1. **No hardware acquisition drivers:** No MAX31855, MAX31856, DS18B20, or DAQ acquisition module is present. Edge checks depend on records being supplied with the relevant raw words/registers, ROM IDs, timing, and event data.
2. **No outbox uploader:** The edge processor appends to a JSONL outbox, but no retrying network sender or outbox acknowledgement/compaction is implemented.
3. **No automatic edge runtime:** The configuration factory exists, but nothing in the repo starts it alongside an acquisition loop. The API server and edge processor are not shown running as one deployed service.
4. **No frontend for new measurement APIs:** The React UI remains the CSV/LSTM experience. Quality flags, physical config editing, and thermal-bias diagnostics are not visible through the current frontend.
5. **CSV path and raw contract differ:** CSV does not contain full hardware provenance, and the loader's quality flags live in its returned summary rather than as `MeasurementRecord` flags in persistent measurement storage.
6. **Malformed CSV rows need model-safe handling:** The CSV loader now preserves missing/nonnumeric cells instead of imputing them. The legacy scaler/model path still consumes the dataframe directly, so such cells can cause scaling or inference failures. A next change should preserve raw columns and make the ML path skip/annotate invalid windows without filling or overwriting sensor values.
7. **Legacy and measurement systems are separate:** Statistical scores in the CSV pipeline are not currently converted into versioned `QualityFlag` records or displayed alongside the newer device API diagnostics.
8. **Hardware behavior is simulated in tests:** Unit tests construct records and register payloads; they do not validate real electrical connections, sensor timing, configuration readback, or physical response.
9. **Engineering parameter quality depends on commissioning:** Tau and other configuration values require measured provenance and reliable operating conditions; software validation cannot establish that a supplied configuration is experimentally correct.
10. **README is behind the code:** The README documents the CSV dashboard and not the newer measurement contract, edge processor, device API, or current limitations.

## 15. Suggested next milestones

1. Define and implement hardware acquisition adapters that create `MeasurementRecord` objects with trustworthy monotonic timing, boot IDs, raw words/registers, and event state.
2. Add a durable outbox sender with retries, acknowledgement, idempotent delivery, and config refresh behavior.
3. Add a CSV model-preparation layer that never edits raw cells and safely skips or marks invalid windows; expose the CSV quality annotations in the result payload and UI.
4. Connect the React interface to measurement, diagnostics, parameter, and physical-configuration endpoints, or document that the hardware path is API-only for the prototype.
5. Run the full pytest suite and frontend build in an environment with the declared dependencies; record versions and results.
6. Validate physical thresholds and bias estimates against commissioning experiments and representative runs. Document the column meanings and provenance of cleaned datasets.
7. Update README and deployment instructions to describe both flows and clearly identify which parts require real acquisition hardware.

## 16. Detailed CSV-to-model execution trace

This section follows the actual CSV request from the HTTP boundary to the UI result.

### 16.1 Upload and file handling

`POST /api/upload` expects a multipart file with a `.csv` filename. The backend saves the uploaded file under its upload directory and returns a generated file identifier and server-side path. `UploadZone.jsx` accepts drag-and-drop or file selection and passes the browser `File` to `uploadCSV()` in `api.js`. `App.jsx` keeps the returned path and later sends that path to `/api/run`.

The current run endpoint accepts a local filesystem path. This is appropriate for the current single-host development setup, where the uploaded file and Flask process share a machine. It is not a portable cloud contract: a remote client should submit an opaque upload ID rather than a server filesystem path. Access control and per-user isolation are not implemented in the code shown here.

### 16.2 Parsing and channel discovery

`load_and_summarize_data()` scans for a line containing one of `Scan Num`, `101 (`, `Scan Swee`, or `timestamp` and treats that line as the tabular header. Pandas reads from that row, trims whitespace from column names, and drops entirely empty columns and rows. It chooses the first column whose name contains `time` or `swee` as the timestamp; when none exists, the row index becomes `Timestamp`.

Sensor discovery is currently heuristic: a column is selected if its name contains uppercase `C` or its lowercase name contains `ch_`. The code excludes `Timestamp` afterward. This recognizes column styles seen in the included files, such as `101 (°C)` and `ch_00`, but it is not a schema-aware parser. Other metadata columns containing `C`, alternate temperature units, duplicate time columns, or mixed files can be misidentified. The `dropna` operations also mean completely blank rows/columns are removed during parsing; these are not treated as sensor samples.

The recent raw-retention change creates numeric temporary series only to find quality conditions. It keeps original dataframe cells while reporting row index, channel, code, classification, and original display value in `summary.quality_flags`. It marks missing values as suspicious and nonnumeric, non-finite, or absolute numeric values over 10,000 as implausible. The `10,000` cutoff is a generic overload heuristic in the CSV loader, not a channel-specific physical limit. The data contract is richer than this CSV summary: CSV annotations are not currently persisted as `QualityFlag` instances.

### 16.3 Data-family classification and metadata

`classify_data_type()` first checks channel naming for `ch_0`, then inspects up to six file-header lines for `daq970a` or `34970a`, then falls back to temperature-column count. This maps uploads to biomass, cataluminescence, default, Swiss Roll, or unknown. It is a project-specific naming/header heuristic rather than a general instrument-identification protocol.

`extract_instrument_metadata()` reads up to 30 initial lines and looks for Address/Model headers. It maps common metadata labels into JSON fields and parses the channel configuration section and module information. This metadata is descriptive: it does not automatically produce hardware `MeasurementRecord` fields or validated physical configurations.

### 16.4 Scaling and sequences

For each channel `j`, `StandardScaler` applies a z-score transform:

```text
z[j] = (x[j] - mean_train[j]) / standard_deviation_train[j]
```

The fitted means/scales are saved and reused with the weights. This prevents a channel with a numerically large unit/range from dominating solely due to magnitude, but it does not make values physically comparable across different probes or calibration conditions.

For `N` rows and a time step of `T=10`, `create_sequences()` generates windows starting at rows 0 through `N-T-1`; each window contains rows `[i, i+T)`. The timestamp assigned to that model score is the last row of the window, at `i+T-1`. Under this implementation there are `N-T` windows when all data are usable. Overlapping windows share nine of ten rows, so neighboring scores are dependent rather than independent observations.

The current loader intentionally stopped imputing missing values. However, `run_pipeline()` still scales the sensor columns and constructs tensors directly. Missing or nonnumeric values can therefore fail scaling, propagate non-finite values into PyTorch, or prevent the model from running. A safe future design should retain the raw dataframe, build a separate numeric model view, mark windows with invalid cells as unscorable, and preserve row-to-window provenance. It should not fill values without a separately approved derived-data policy.

### 16.5 Autoencoder computation in more detail

At timestep `t`, an LSTM receives channel vector `x_t` and previous hidden/cell state. Its gates are functions of current input and prior hidden state. A common LSTM form is:

```text
i_t = sigmoid(W_i x_t + U_i h_(t-1) + b_i)
f_t = sigmoid(W_f x_t + U_f h_(t-1) + b_f)
g_t = tanh   (W_g x_t + U_g h_(t-1) + b_g)
o_t = sigmoid(W_o x_t + U_o h_(t-1) + b_o)
c_t = f_t * c_(t-1) + i_t * g_t
h_t = o_t * tanh(c_t)
```

Here `i_t` controls new information entering memory, `f_t` controls retention/forgetting of prior cell state, `g_t` is candidate content, and `o_t` controls exposed hidden output. PyTorch supplies this gated recurrence; the project specifies the layers and input/output dimensions rather than hand-implementing gates.

The encoder processes the ten-row multichannel window and returns the final hidden state `h_T`. The code takes the last layer's hidden state, repeats it ten times, and supplies that repeated vector sequence to the decoder LSTM. A linear layer maps each decoder state back to `F` channels. This is a sequence-to-sequence reconstruction with a fixed-length latent summary. Repeating one encoding at each decoder input forces the decoder to reconstruct the temporal pattern from the summary and learned recurrent dynamics.

The training loss for batch `B`, time steps `T`, and channels `F` is:

```text
L = (1 / (B*T*F)) * sum over b,t,f of abs(x[b,t,f] - x_hat[b,t,f])
```

The optimizer computes gradients of this loss through both recurrent layers and updates weights using Adam. In feedback training, a sample-level weight `w_b` is applied to each window's elementwise loss before averaging. Windows whose final timestamp string matches a researcher-confirmed normal timestamp get `w_b=2`; other windows get `1`. Exact timestamp formatting/matching matters, and confirmed anomaly labels are stored but are not used as a direct positive training target in this loss.

### 16.6 Scoring, KDE, thresholding, and display confidence

For window `i`, the implementation computes:

```text
MAE_i = (1 / (T*F)) * sum over t,f of abs(x_i[t,f] - x_hat_i[t,f])
```

It also calculates a per-channel error vector from the last timestep of every window. These feature-level values support a contribution chart, but they do not assign a causal explanation to a sensor.

The backend estimates the density of MAE scores with `scipy.stats.gaussian_kde`. It evaluates the density at 1,000 evenly spaced score values between the minimum and maximum. If `argrelextrema` finds one or more local minima, the first minimum is used as the threshold. This rule assumes the error distribution has a meaningful valley; the first valley can instead be a small local fluctuation. If no valley exists, the 97th percentile is used. Neither threshold rule has a calibrated error rate without labeled validation runs.

On first training, the threshold is estimated from reconstruction errors on the training windows and saved. On a model-cache hit, that saved threshold is used for anomaly decisions. The current-run KDE is nevertheless calculated for the chart, so the visible density curve and the saved decision threshold can differ. KDE fitting can also be ill-conditioned if there are too few scores or nearly constant scores; this case is not comprehensively guarded in the pipeline.

An anomaly is `MAE > threshold`. The returned “confidence” is a display transformation, not statistical confidence:

```text
confidence = 0,                         when MAE <= threshold
confidence = 50 + 50*(MAE-threshold)/(max_MAE-threshold), otherwise
```

The formula is capped at 100, and the maximum scored point reaches 100. As a result, confidence depends on the other scores in the same run and can change when the dataset changes, even when a particular point's MAE does not. It is not a calibrated probability.

### 16.7 Result statistics and plots

The results builder creates per-channel arrays with timestamps, raw plotted values, anomaly timestamps/values, and display confidence. It also returns a 50-bin MAE histogram, KDE curve and threshold details, training loss per epoch, anomaly table, and last-timestep feature errors.

Per-channel statistics include mean, standard deviation, min/max, quartiles, IQR, skewness, kurtosis, and a nominal 95% interval calculated as `mean +/- 1.96 * standard_error`. This is a normal-approximation interval, not a small-sample Student-t interval, and serial correlation between sensor readings is not accounted for. The effective uncertainty can consequently be understated for autocorrelated data.

## 17. Backend module-by-module technical reference

### 17.1 `app.py`: orchestration and HTTP boundary

`app.py` is both the legacy model implementation and the Flask API module. It initializes upload, feedback, model, and dashboard-store directories; registers Flask-CORS; and installs a JSON provider that converts NumPy scalars/arrays and non-finite floats into JSON-safe values. The source also has recursive `sanitize()` for result dictionaries.

The file contains model key/artifact helpers, threshold estimation, model save/load/list/delete operations, SQLite feedback functions, data classification, instrument metadata parsing, model architecture/training/scoring, and all Flask routes. This centralization is convenient for the prototype but couples HTTP handling, file processing, ML, and persistence. Splitting route/service/model layers would make future tests and runtime boundaries clearer.

Route behavior in the current code:

| Route | Inputs and outputs / behavior |
|---|---|
| `GET /api/health` | Lightweight health response |
| `POST /api/upload` | Saves a CSV upload and returns its server path/identifier |
| `POST /api/run` | Runs the model pipeline for a path and optional feedback setting; returns run ID and result payload |
| `POST /api/feedback` | Persists reviewer labels and sample/model context |
| `GET /api/feedback/<data_type>` | Lists stored labels for a family |
| `GET /api/feedback/summary` | Counts feedback by family and label |
| `GET /api/results/<run_id>` | Returns in-memory cached result for a run ID |
| `POST /api/measurements` | Validates versioned records and persists them, returning inserted and duplicate counts |
| `GET /api/measurements/<device_id>` | Returns complete stored raw records, optionally filtered by channel |
| `GET /api/measurements/<device_id>/science` | Returns the science-filtered record collection |
| `GET /api/diagnostics/<device_id>` | Returns aggregate quality and diagnostic annotations |
| `GET /api/analytics/<device_id>/parameters` | Returns channel parameter estimates and sample inclusion counts |
| `GET/PUT /api/edge-config/<device_id>` | Reads or stores a versioned generic edge configuration bundle |
| `GET/PUT /api/devices/<device_id>/channels/<channel_id>/physical-config` | Reads or validates/publishes one channel physical config |
| `GET /api/devices/<device_id>/physical-config` | Lists latest configs for a device |
| `POST /api/retrain` | Runs forced retraining with feedback mode enabled |
| `GET /api/models` | Lists model metadata and artifact existence |
| `DELETE /api/models/<model_key>` | Deletes the weights, scaler, and metadata artifacts |

Run results are stored in an in-memory dictionary, so they do not survive a server restart. The CSV endpoint passes a server filesystem path. The module's `__main__` starts Flask in debug mode on port 5000; production deployment should use a production WSGI server and disable debug mode.

### 17.2 `measurement_contract.py`: validation and transport model

Besides `MeasurementRecord`, this file defines `SensorType`, `FrontEndType`, `QualityClass`, `QualityFlag`, `DiagnosticAnnotation`, `EventMetadata`, `TimingMetadata`, and `IntervalStatistics`.

The record is frozen (`ConfigDict(frozen=True)`) and rejects unexpected fields (`extra="forbid"`), making schema mistakes visible at ingestion. Timestamps require timezone awareness. Monotonic uptime/sample interval have nonnegative/positive bounds. Conversion completion cannot precede conversion start. Quality codes must be unique within one record. `EventMetadata` allows extra fields to accommodate event extensions, while core timing and quality models are stricter.

The contract currently does not validate every physical relationship: it does not, for example, enforce that `sensor_type` necessarily matches `front_end_type`, that `raw_value` is finite, or that a record's version strings match the active config. Those are potential future validation rules. Raw register mappings are flexible by design because the hardware payload differs between front-ends.

### 17.3 `edge_layer.py`: signal checks, state transitions, and outbox

`EdgeStateStore` creates a SQLite `edge_state` table keyed by `(device_id, channel_id)`. It serializes state JSON, commits each update, and protects writes/outbox appends with a thread lock. `enqueue()` appends a serialized full record to a JSON Lines file and flushes Python's file buffer. It does not `fsync`, mark delivery, or compact acknowledged rows; durability/retry semantics beyond local append remain future work.

`EdgeProcessor.process()` uses this sequence:

1. Load persisted per-channel detector state.
2. Copy existing quality flags and add timing flags.
3. Add front-end integrity flags.
4. If a config is installed for this channel, run the physics detector and thermal-bias analyzer.
5. Deduplicate flags by code and enforce statistical flag authority.
6. Create an updated immutable record; raw value/hardware data remain from the input.
7. Update counters, boot/timing and integrity state; persist state; append output record to the outbox.

The new config loader stores a map keyed by channel ID and validates that each mapping key matches the config's own `channel_id`. It does not currently reject a config whose `device_id` differs from the processor's intended device, because the processor itself is not bound to one device. Multi-device deployments should use stronger identity scoping.

Detailed checks:

| Front-end | Implementation behavior | Required input from acquisition |
|---|---|---|
| MAX31855 | `MAX31855_RESERVED_MASK` checks bits 17 and 3; catches `0` / `0xffffffff`; compares duplicate reads if supplied; checks summary-vs-specific fault agreement | `raw_hardware_word`; optional duplicate word(s), summary fault, open, short-to-VCC, short-to-GND |
| MAX31856 | Creates separate flags for named fault conditions; checks config readback/reset and optional expected-vs-actual mapping; bounds thermocouple/CJ temperatures | Fault booleans, expected/readback config, reset indication, thermocouple and CJ temperatures |
| DS18B20 | CRC-8 uses reflected polynomial `0x8c`; checks raw word, CRC, bus presence, ROM mismatch, topology and 85 C conversion state; periodically compares whole bus manifest | Scratchpad bytes/CRC result, ROM IDs, bus IDs, conversion state, configured manifest |

Timing logic compares within-boot monotonic deltas to declared sample interval using a tolerance of the larger of 0.05 seconds or 25% of the declared interval. It flags non-advancing monotonic uptime as implausible. A boot ID change is suspicious because state continuity crosses a restart. Conversion delay is compared to measured start/completion with a 0.01-second tolerance.

State changes carry across restarts in SQLite. One implementation detail to validate under concurrency is that the SQLite connection is shared across threads with `check_same_thread=False`, but only write operations are locked; reads are not serialized by the lock.

### 17.4 `physics_detector.py`: equations and state

The detector order is deliberate: hardware/timing checks happen first in `EdgeProcessor`, followed by bounds, slope, cold-soak, high-pass energy, cross-sensor lag, tau drift, and gated drift.

For physical bounds, the allowed interval is:

```text
lower = max(minimum_physical_temperature, minimum_sink_temperature)
upper = min(maximum_physical_temperature, maximum_source_temperature)
```

For slope, the measured rate is `abs(T_current - T_previous) / delta_t`. The implementation approximates the permitted rate using the largest distance from the current temperature to configured maximum-source/minimum-sink temperatures, divided by tau, then expands by relative tau uncertainty:

```text
slope_limit = drive_gap / tau * (1 + tau_uncertainty / tau)
```

This is a conservative coded approximation to a first-order thermal model, not a full source trajectory estimator. It uses configured extrema rather than a measured instantaneous drive temperature.

Cold-soak agreement uses the greater of three times the configured noise floor and three times datasheet resolution as the tolerance. High-pass processing stores up to 20 successive first differences, computes RMS, and compares to the greater of three times noise floor or three times resolution. Its message describes a cutoff derived from `10 * (1/(2*pi*tau))`; it is not a general FFT spectrum estimator. A precomputed `high_pass_rms` can be combined by taking the larger of supplied and computed RMS.

Peer checks calculate the absolute difference between the sample's uptime and peer uptime after subtracting the expected offset. The timing tolerance is the larger of configured sample interval and noise-floor numeric value; those two terms have different physical units (seconds versus temperature), which should be revisited before using this threshold as a calibrated rule. A residual check compares temperatures using a configured scale and only runs if related sensor pairs are configured.

The online tau estimate uses a lag-one correlation over a short history. For correlation `rho` in `(0,1)`, it calculates `tau_hat = -sample_interval / ln(rho)`. Drift is flagged when the difference from configured tau exceeds the larger of three tau-uncertainties or 25% of configured tau. This estimate assumes an approximately first-order response and meaningful transient excitation; stable/noisy steady-state data may not identify tau.

Gated drift updates only if `event_metadata.attributes.quiescent_window` is true and no implausible flag exists. The one-sided accumulation is `CUSUM = max(0, previous_CUSUM + current - baseline - noise_floor)`. A threshold of five times the larger configured noise floor/resolution triggers a suspicious flag. This form emphasizes upward drift; it is not a two-sided change detector.

### 17.5 `physical_configuration.py`: commissioning constraints

The configuration model treats measured tau and channel identity as essential physical context. It rejects catalogue-only tau provenance by restricting `tau_measurement_method` to commissioning step, online transient fit, or both. It also enforces `tau_uncertainty < tau` so the uncertainty does not exceed the measured characteristic response time.

`SensorPair` provides a peer channel and `fast`/`slow` relationship plus expected time offset. `metadata` can hold extension values used by thermal diagnostics such as emissivity or recovery factor. The model itself validates structure and selected bounds; it cannot verify commissioning procedures, board identity, insertion-depth truth, sensor placement, or actual tau measurement quality.

### 17.6 `thermal_bias.py`: diagnostic calculations

The analyzer is called after physics checks only when a channel config is installed. It reads optional environmental/installation metadata from `raw_register_data` and emits typed annotations.

- **Radiation:** `bias = emissivity * sigma * (T_probe_K^4 - T_wall_K^4) / h`. Temperatures are converted from Celsius to Kelvin; nonpositive `h` disables the estimate. This estimates the temperature offset that would balance the coded radiation exchange against convective heat transfer under simplified assumptions.
- **Conduction:** derives depth/diameter ratio and an exponential attenuation `exp(-ratio/10)`, then scales the mount-to-probe difference. Ratios at least 20 suppress the annotation. The attenuation relationship is a heuristic in code and requires validation for probe geometry/material/flow.
- **Recovery:** above 50 m/s, estimates `r * velocity^2 / (2 * cp)`, using supplied recovery factor and heat capacity or metadata/defaults. This has a high fixed speed gate and relies on gas properties and probe recovery factor being supplied appropriately.
- **Inhomogeneity:** emits a qualitative annotation only when repositioning/signature/temperature-band indicators are explicitly true.
- **Cold junction:** annotates when supplied board temperature changes by at least 0.5 C from the previous sample in persisted detector state.
- **MAX31855 linearization:** annotates a supplied difference against a reference front-end; it does not independently calculate MAX31855-vs-MAX31856 calibration error.
- **Junction style:** emits contextual annotation for every configured junction style other than not-applicable.

The analyzer records the last thermal-bias value in state, but the current formulas do not correct readings, estimate every effect from temperature alone, or automatically decide whether the reading is faulty.

### 17.7 `quality_policy.py`: suppression authority

`is_statistical_detector()` recognizes a fixed set of statistical detector names and any detector name containing terms such as `kde`, `reconstruction`, `statistical`, or `autoencoder`. `enforce_flag_authority()` downgrades such flags to `unusual`. The policy is name-based, so detector naming conventions matter; unknown statistical detector names that omit these tokens could bypass normalization.

`quality_summary()` counts flags by class and reports if science exclusion applies. `science_view()` is a pure filter over immutable records and returns records without any implausible flag. It does not mutate the raw store. Suspicious and unusual records stay in the view.

### 17.8 `dashboard_store.py`: SQLite persistence and analytics

`DashboardStore` creates tables for measurements, generic versioned edge configs, and physical channel configs. Measurement rows have an indexed device/channel/timestamp key path. `measurement_id` is the primary key; `INSERT OR IGNORE` makes repeated IDs idempotent from the store's perspective. The full payload JSON is stored alongside indexed scalar columns.

Diagnostics traverses retrieved records and flattens each quality flag and diagnostic annotation with measurement ID, channel, and timestamp. Quality summaries are totals of flags, not unique affected-record counts. Parameter fitting groups raw and science records by channel, computes arithmetic mean, estimates a noise scale from adjacent differences (`RMS(diff - mean_value) / sqrt(2)` in the current implementation), and estimates tau from lag-one correlation when possible. The implemented noise calculation's centering is worth reviewing: first differences are centered using the level mean rather than the difference mean, so it is not a standard noise estimator as currently written. Also, `list_measurements()` defaults to a 1,000-record limit, and parameter fitting uses that default; large histories therefore need pagination/aggregation before fitted values represent all stored data.

Configuration version retrieval supports a requested version or the newest saved version. The list endpoint returns the latest configuration per channel. Generic edge configuration and per-channel physical configuration are separate tables and APIs; there is no automatic transformation of fitted parameters into a new physical config version.

### 17.9 Frontend component roles

The UI is a single-page React app built with Vite and Recharts:

| Component | Role |
|---|---|
| `App.jsx` | Upload/run/rerun/reset state, feedback toggle, result state and layout |
| `api.js` | Browser HTTP client for CSV and feedback routes |
| `UploadZone.jsx` | Drag/drop and file picker |
| `ResultsDashboard.jsx` | Tabs and composition of result views |
| `ChannelChart.jsx` | Per-channel line chart, anomaly scatter overlay, tooltip; downsamples long series while retaining anomaly points |
| `AnomalyTable.jsx` | Sorts/pages candidates and gathers Yes/No feedback |
| `MetadataPanel.jsx` | Displays instrument metadata and channel config extracted from file headers |
| `StatisticsPanel.jsx` | Displays descriptive statistics and charts for skew/kurtosis/confidence intervals |
| `TrainingLossChart.jsx` | Plots epoch loss |
| `MAEDistributionChart.jsx` | Plots reconstruction error distribution/threshold |
| `KDEChart.jsx` | Plots estimated KDE curve and threshold context |
| `ConfidenceDistributionChart.jsx` | Displays display-confidence distribution |
| `FeatureContributionChart.jsx` | Displays per-channel reconstruction error contribution |

The UI's “confidence” terminology is inherited from the legacy model presentation. As described above, the backend number is a score rescaling and not probability calibration. The modern measurement diagnostics APIs have no corresponding React components or API client calls yet.

### 17.10 Dependencies and runtime responsibilities

`requirements.txt` declares Flask/Flask-CORS, Pydantic, typing extensions, NumPy, pandas, scikit-learn, SciPy, and PyTorch. These support HTTP, contract validation, CSV/table processing, scaling, KDE/local minima, and sequence modeling. The frontend package declares React/Vite/TypeScript and Recharts. SQLite is built into Python and is used without an external server.

The repo's `package-lock.json` exists at both dashboard frontend and root level, but only the frontend package contains the visible Vite scripts. Python setup instructions in README use pip and `python app.py`; in the audit environment no Python executable/launcher was available, so tests were not run in that environment.

## 18. Mathematical and scientific assumptions to validate

The learned model and physics detector answer different questions. The autoencoder estimates whether a pattern resembles training windows. The physical detector tests specific engineering constraints. A physically valid but rare startup may have high reconstruction error and should remain an unusual observation; a physically impossible reading can be implausible even if the learned model reconstructs it well.

The `tau` model assumes a lumped first-order sensor response. In continuous time:

```text
dT/dt = (T_infinity - T) / tau
```

For constant `T_infinity`, its solution is `T(t) = T_infinity + (T(0)-T_infinity) exp(-t/tau)`. Tau is the characteristic time in which the remaining temperature difference decays to `1/e` of its initial value. The response depends on junction style, probe dimensions, insertion depth, material, installation, gas flow, and surrounding medium; this is why config validation asks for a measured commissioning value.

The CSV model's z-score scaling and global MAE threshold assume the channel order and training distribution remain representative. By contrast, physics bounds assume the configured source/sink limits and installation data accurately represent the process. Both systems can be wrong if their input assumptions are wrong. Version fields, raw payloads, event context, config provenance, and stored thresholds are intended to support later audit; the current CSV model metadata does not yet persist every preprocessing/runtime dependency or a complete dataset fingerprint.

Thermal-bias estimates also depend on assumptions. The radiation expression balances a simplified radiative term against a supplied heat-transfer coefficient; conduction uses a heuristic attenuation; recovery uses a supplied factor and heat capacity. These are explanatory models and should be calibrated against commissioning comparisons, not interpreted as metrological corrections without validation.

## 19. Test inventory and evidence boundary

The test files correspond to these behaviors:

| Test file | What it exercises |
|---|---|
| `test_measurement_contract.py` | Record fields, additive flags, duplicate-code rejection, and timing validation |
| `test_timing.py` | Restart persistence, monotonic regressions, interval mismatch, conversion timing, oversample summaries |
| `test_signal_integrity.py` | MAX31855 words/fault consistency, MAX31856 faults/config/limits, DS18B20 CRC/ROM/topology/rate |
| `test_physical_configuration.py` | Required measured tau provenance, validation, persistence/retrieval |
| `test_physical_configuration_api.py` | Config PUT/GET/list and edge processor config loading |
| `test_physics_detector.py` | Bounds, slope, cold-soak, high-pass, peer lag, tau fit, gated drift |
| `test_thermal_bias.py` | Expected effects are diagnostics and serialize separately from flags |
| `test_quality_policy.py` | Statistical downgrading, science view, raw retention, and fit exclusions |
| `test_measurement_api.py` | Measurement ingestion/retrieval and edge config route behavior |
| `test_layers.py` | Edge flags/state/outbox, dashboard storage, duplicate ingestion, and configuration bundles |

These are unit/integration tests using constructed inputs and temporary SQLite databases. They do not demonstrate performance on a real acquisition stream, held-out experimental validity, hardware-driver correctness, network resilience, security, or UI use of the new APIs. The recent code-review session could not execute them because Python was unavailable; the repository's existence of tests should not be reported as a successful test run.

## 20. Conclusion

The repository has evolved from an offline, researcher-facing CSV anomaly dashboard into a codebase that also contains a structured measurement-quality architecture. The LSTM/KDE and feedback workflow are implemented as a dashboard path. The measurement contract, edge checks, physics annotations, thermal-bias diagnostics, quality authority, persistence, APIs, and science-only parameter fitting are implemented as backend modules with focused tests.

The principal remaining gap is integration: hardware acquisition, offline-to-cloud delivery, automatic edge startup/config refresh, and frontend diagnostics have not yet connected the newer modules into a deployed end-to-end sensor system. The LSTM autoencoder is a useful unsupervised baseline, while the physics and signal-integrity modules add interpretable constraints. Neither model training artifacts nor unit tests alone establish field accuracy. The project should be described as a substantial software prototype whose hardware-dependent assumptions and end-to-end operation still require execution, commissioning, and validation.
