---
base_model: sentence-transformers/all-mpnet-base-v2
library_name: sentence-transformers
pipeline_tag: sentence-similarity
tags:
- sentence-transformers
- sentence-similarity
- feature-extraction
- generated_from_trainer
- dataset_size:10432
- loss:MultipleNegativesRankingLoss
widget:
- source_sentence: NightClub has chosen file names to appear legitimate including
    EsetUpdate-0117583943.exe for its dropper.
  sentences:
  - 'T1003.001: Adversaries may attempt to access credential material stored in the
    process memory of the Local Security Authority Subsystem Service (LSASS). After
    a user logs on, the system generates and stores a variety of credential materials
    in LSASS process memory. These credential materials can be harvested by an administrative
    user or SYSTEM and used to conduct [Lateral Movement](https://attack.mitre.org/tactics/TA0008)
    using [Use Alternate Authentication Material](https://attack.mitre.org/techniques/T1550).
    As well as in-memory techniques, the LSASS process memory can be dumped from the
    target host and analyzed on a local system. For example, on the target host use
    procdump: * <code>procdump -ma lsass.exe lsass_dump</code> Locally, mimikatz can
    be run using: * <code>sekurlsa::Minidump lsassdump.dmp</code> * <code>sekurlsa::logonPasswords</code>
    Built-in Windows tools such as `comsvcs.dll` can also be used: * <code>rundll32.exe
    C:\Windows\System32\comsvcs.dll MiniDump PID lsass.dmp full</code> Similar to
    [Image File Execution Options Injection](https://attack.mitre.org/techniques/T1546/012),
    the silent process exit mechanism can be abused to create a memory dump of `lsass.exe`
    through Windows Error Reporting (`WerFault.exe`). Windows Security Support Provider
    (SSP) DLLs are loaded into LSASS process at system start. Once loaded into the
    LSA, SSP DLLs have access to encrypted and plaintext passwords that are stored
    in Windows, such as any logged-on user''s Domain password or smart card PINs.
    The SSP configuration is stored in two Registry keys: <code>HKLM\SYSTEM\CurrentControlSet\Control\Lsa\Security
    Packages</code> and <code>HKLM\SYSTEM\CurrentControlSet\Control\Lsa\OSConfig\Security
    Packages</code>. An adversary may modify these Registry keys to add new SSPs,
    which will be loaded the next time the system boots, or when the AddSecurityPackage
    Windows API function is called. The following SSPs can be used to access credentials:
    * Msv: Interactive logons, batch logons, and service logons are done through the
    MSV authentication package. * Wdigest: The Digest Authentication protocol is designed
    for use with Hypertext Transfer Protocol (HTTP) and Simple Authentication Security
    Layer (SASL) exchanges. * Kerberos: Preferred for mutual client-server domain
    authentication in Windows 2000 and later. * CredSSP: Provides SSO and Network
    Level Authentication for Remote Desktop Services.'
  - 'T1036.005: Adversaries may match or approximate the name or location of legitimate
    files, Registry keys, or other resources when naming/placing them. This is done
    for the sake of evading defenses and observation. This may be done by placing
    an executable in a commonly trusted directory (ex: under System32) or giving it
    the name of a legitimate, trusted program (ex: `svchost.exe`). Alternatively,
    a Windows Registry key may be given a close approximation to a key used by a legitimate
    program. In containerized environments, a threat actor may create a resource in
    a trusted namespace or one that matches the naming convention of a container pod
    or cluster.'
  - 'T1071.001: Adversaries may communicate using application layer protocols associated
    with web traffic to avoid detection/network filtering by blending in with existing
    traffic. Commands to the remote system, and often the results of those commands,
    will be embedded within the protocol traffic between the client and server. Protocols
    such as HTTP/S and WebSocket that carry web traffic may be very common in environments.
    HTTP/S packets have many fields and headers in which data can be concealed. An
    adversary may abuse these protocols to communicate with systems under their control
    within a victim network while also mimicking normal, expected traffic.'
- source_sentence: Files on various operating systems can have a complex format which
    allows for the storage of other data, in addition to its contents. Often this
    is metadata about the file, such as a cached thumbnail for an image file. Unless
    utilities are invoked in a particular way, this data is not visible during the
    normal use of the file. It is possible for an attacker to store malicious data
    or code using these facilities, which would be difficult to discover.
  sentences:
  - 'T1552.004: Adversaries may search for private key certificate files on compromised
    systems for insecurely stored credentials. Private cryptographic keys and certificates
    are used for authentication, encryption/decryption, and digital signatures. Common
    key and certificate file extensions include: .key, .pgp, .gpg, .ppk., .p12, .pem,
    .pfx, .cer, .p7b, .asc. Adversaries may also look in common key directories, such
    as <code>~/.ssh</code> for SSH keys on * nix-based systems or <code>C:&#92;Users&#92;(username)&#92;.ssh&#92;</code>
    on Windows. Adversary tools may also search compromised systems for file extensions
    relating to cryptographic keys and certificates. When a device is registered to
    Entra ID, a device key and a transport key are generated and used to verify the
    device’s identity. An adversary with access to the device may be able to export
    the keys in order to impersonate the device. On network devices, private keys
    may be exported via [Network Device CLI](https://attack.mitre.org/techniques/T1059/008)
    commands such as `crypto pki export`. Some private keys require a password or
    passphrase for operation, so an adversary may also use [Input Capture](https://attack.mitre.org/techniques/T1056)
    for keylogging or attempt to [Brute Force](https://attack.mitre.org/techniques/T1110)
    the passphrase off-line. These private keys can be used to authenticate to [Remote
    Services](https://attack.mitre.org/techniques/T1021) like SSH or for use in decrypting
    other collected files such as email.'
  - 'T1059.003: Adversaries may abuse the Windows command shell for execution. The
    Windows command shell ([cmd](https://attack.mitre.org/software/S0106)) is the
    primary command prompt on Windows systems. The Windows command prompt can be used
    to control almost any aspect of a system, with various permission levels required
    for different subsets of commands. The command prompt can be invoked remotely
    via [Remote Services](https://attack.mitre.org/techniques/T1021) such as [SSH](https://attack.mitre.org/techniques/T1021/004).
    Batch files (ex: .bat or .cmd) also provide the shell with a list of sequential
    commands to run, as well as normal scripting operations such as conditionals and
    loops. Common uses of batch files include long or repetitive tasks, or the need
    to run the same set of commands on multiple systems. Adversaries may leverage
    [cmd](https://attack.mitre.org/software/S0106) to execute various commands and
    payloads. Common uses include [cmd](https://attack.mitre.org/software/S0106) to
    execute a single command, or abusing [cmd](https://attack.mitre.org/software/S0106)
    interactively with input and output forwarded over a command and control channel.'
  - 'T1218.001: Adversaries may abuse Compiled HTML files (.chm) to conceal malicious
    code. CHM files are commonly distributed as part of the Microsoft HTML Help system.
    CHM files are compressed compilations of various content such as HTML documents,
    images, and scripting/web related programming languages such VBA, JScript, Java,
    and ActiveX. CHM content is displayed using underlying components of the Internet
    Explorer browser loaded by the HTML Help executable program (hh.exe). A custom
    CHM file containing embedded payloads could be delivered to a victim then triggered
    by [User Execution](https://attack.mitre.org/techniques/T1204). CHM execution
    may also bypass application application control on older and/or unpatched systems
    that do not account for execution of binaries through hh.exe.'
- source_sentence: PowGoop can use a modified Base64 encoding mechanism to send data
    to and from the C2 server.
  sentences:
  - 'T1547.001: Adversaries may achieve persistence by adding a program to a startup
    folder or referencing it with a Registry run key. Adding an entry to the "run
    keys" in the Registry or startup folder will cause the program referenced to be
    executed when a user logs in. These programs will be executed under the context
    of the user and will have the account''s associated permissions level. The following
    run keys are created by default on Windows systems: * <code>HKEY_CURRENT_USER\Software\Microsoft\Windows\CurrentVersion\Run</code>
    * <code>HKEY_CURRENT_USER\Software\Microsoft\Windows\CurrentVersion\RunOnce</code>
    * <code>HKEY_LOCAL_MACHINE\Software\Microsoft\Windows\CurrentVersion\Run</code>
    * <code>HKEY_LOCAL_MACHINE\Software\Microsoft\Windows\CurrentVersion\RunOnce</code>
    Run keys may exist under multiple hives. The <code>HKEY_LOCAL_MACHINE\Software\Microsoft\Windows\CurrentVersion\RunOnceEx</code>
    is also available but is not created by default on Windows Vista and newer. Registry
    run key entries can reference programs directly or list them as a dependency.
    For example, it is possible to load a DLL at logon using a "Depend" key with RunOnceEx:
    <code>reg add HKLM\SOFTWARE\Microsoft\Windows\CurrentVersion\RunOnceEx\0001\Depend
    /v 1 /d "C:\temp\evil[.]dll"</code> Placing a program within a startup folder
    will also cause that program to execute when a user logs in. There is a startup
    folder location for individual user accounts as well as a system-wide startup
    folder that will be checked regardless of which user account logs in. The startup
    folder path for the current user is <code>C:\Users\\[Username]\AppData\Roaming\Microsoft\Windows\Start
    Menu\Programs\Startup</code>. The startup folder path for all users is <code>C:\ProgramData\Microsoft\Windows\Start
    Menu\Programs\StartUp</code>. The following Registry keys can be used to set startup
    folder items for persistence: * <code>HKEY_CURRENT_USER\Software\Microsoft\Windows\CurrentVersion\Explorer\User
    Shell Folders</code> * <code>HKEY_CURRENT_USER\Software\Microsoft\Windows\CurrentVersion\Explorer\Shell
    Folders</code> * <code>HKEY_LOCAL_MACHINE\SOFTWARE\Microsoft\Windows\CurrentVersion\Explorer\Shell
    Folders</code> * <code>HKEY_LOCAL_MACHINE\SOFTWARE\Microsoft\Windows\CurrentVersion\Explorer\User
    Shell Folders</code> The following Registry keys can control automatic startup
    of services during boot: * <code>HKEY_LOCAL_MACHINE\Software\Microsoft\Windows\CurrentVersion\RunServicesOnce</code>
    * <code>HKEY_CURRENT_USER\Software\Microsoft\Windows\CurrentVersion\RunServicesOnce</code>
    * <code>HKEY_LOCAL_MACHINE\Software\Microsoft\Windows\CurrentVersion\RunServices</code>
    * <code>HKEY_CURRENT_USER\Software\Microsoft\Windows\CurrentVersion\RunServices</code>
    Using policy settings to specify startup programs creates corresponding values
    in either of two Registry keys: * <code>HKEY_LOCAL_MACHINE\Software\Microsoft\Windows\CurrentVersion\Policies\Explorer\Run</code>
    * <code>HKEY_CURRENT_USER\Software\Microsoft\Windows\CurrentVersion\Policies\Explorer\Run</code>
    Programs listed in the load value of the registry key <code>HKEY_CURRENT_USER\Software\Microsoft\Windows
    NT\CurrentVersion\Windows</code> run automatically for the currently logged-on
    user. By default, the multistring <code>BootExecute</code> value of the registry
    key <code>HKEY_LOCAL_MACHINE\System\CurrentControlSet\Control\Session Manager</code>
    is set to <code>autocheck autochk *</code>. This value causes Windows, at startup,
    to check the file-system integrity of the hard disks if the system has been shut
    down abnormally. Adversaries can add other programs or processes to this registry
    value which will automatically launch at boot. Adversaries can use these configuration
    locations to execute malware, such as remote access tools, to maintain persistence
    through system reboots. Adversaries may also use [Masquerading](https://attack.mitre.org/techniques/T1036)
    to make the Registry entries look as if they are associated with legitimate programs.'
  - 'T1132.002: Adversaries may encode data with a non-standard data encoding system
    to make the content of command and control traffic more difficult to detect. Command
    and control (C2) information can be encoded using a non-standard data encoding
    system that diverges from existing protocol specifications. Non-standard data
    encoding schemes may be based on or related to standard data encoding schemes,
    such as a modified Base64 encoding for the message body of an HTTP request.'
  - 'T1574.007: Adversaries may execute their own malicious payloads by hijacking
    environment variables used to load libraries. The PATH environment variable contains
    a list of directories (User and System) that the OS searches sequentially through
    in search of the binary that was called from a script or the command line. Adversaries
    can place a malicious program in an earlier entry in the list of directories stored
    in the PATH environment variable, resulting in the operating system executing
    the malicious binary rather than the legitimate binary when it searches sequentially
    through that PATH listing. For example, on Windows if an adversary places a malicious
    program named "net.exe" in `C:\example path`, which by default precedes `C:\Windows\system32\net.exe`
    in the PATH environment variable, when "net" is executed from the command-line
    the `C:\example path` will be called instead of the system''s legitimate executable
    at `C:\Windows\system32\net.exe`. Some methods of executing a program rely on
    the PATH environment variable to determine the locations that are searched when
    the path for the program is not given, such as executing programs from a [Command
    and Scripting Interpreter](https://attack.mitre.org/techniques/T1059). Adversaries
    may also directly modify the $PATH variable specifying the directories to be searched.
    An adversary can modify the `$PATH` variable to point to a directory they have
    write access. When a program using the $PATH variable is called, the OS searches
    the specified directory and executes the malicious binary. On macOS, this can
    also be performed through modifying the $HOME variable. These variables can be
    modified using the command-line, launchctl, [Unix Shell Configuration Modification](https://attack.mitre.org/techniques/T1546/004),
    or modifying the `/etc/paths.d` folder contents.'
- source_sentence: SILENTTRINITY can use `cmd.exe` to enable lateral movement using
    DCOM.
  sentences:
  - 'T1059.003: Adversaries may abuse the Windows command shell for execution. The
    Windows command shell ([cmd](https://attack.mitre.org/software/S0106)) is the
    primary command prompt on Windows systems. The Windows command prompt can be used
    to control almost any aspect of a system, with various permission levels required
    for different subsets of commands. The command prompt can be invoked remotely
    via [Remote Services](https://attack.mitre.org/techniques/T1021) such as [SSH](https://attack.mitre.org/techniques/T1021/004).
    Batch files (ex: .bat or .cmd) also provide the shell with a list of sequential
    commands to run, as well as normal scripting operations such as conditionals and
    loops. Common uses of batch files include long or repetitive tasks, or the need
    to run the same set of commands on multiple systems. Adversaries may leverage
    [cmd](https://attack.mitre.org/software/S0106) to execute various commands and
    payloads. Common uses include [cmd](https://attack.mitre.org/software/S0106) to
    execute a single command, or abusing [cmd](https://attack.mitre.org/software/S0106)
    interactively with input and output forwarded over a command and control channel.'
  - 'T1036.005: Adversaries may match or approximate the name or location of legitimate
    files, Registry keys, or other resources when naming/placing them. This is done
    for the sake of evading defenses and observation. This may be done by placing
    an executable in a commonly trusted directory (ex: under System32) or giving it
    the name of a legitimate, trusted program (ex: `svchost.exe`). Alternatively,
    a Windows Registry key may be given a close approximation to a key used by a legitimate
    program. In containerized environments, a threat actor may create a resource in
    a trusted namespace or one that matches the naming convention of a container pod
    or cluster.'
  - 'T1547.012: Adversaries may abuse print processors to run malicious DLLs during
    system boot for persistence and/or privilege escalation. Print processors are
    DLLs that are loaded by the print spooler service, `spoolsv.exe`, during boot.
    Adversaries may abuse the print spooler service by adding print processors that
    load malicious DLLs at startup. A print processor can be installed through the
    <code>AddPrintProcessor</code> API call with an account that has <code>SeLoadDriverPrivilege</code>
    enabled. Alternatively, a print processor can be registered to the print spooler
    service by adding the <code>HKLM\SYSTEM\\[CurrentControlSet or ControlSet001]\Control\Print\Environments\\[Windows
    architecture: e.g., Windows x64]\Print Processors\\[user defined]\Driver</code>
    Registry key that points to the DLL. For the malicious print processor to be correctly
    installed, the payload must be located in the dedicated system print-processor
    directory, that can be found with the <code>GetPrintProcessorDirectory</code>
    API call, or referenced via a relative path from this directory. After the print
    processors are installed, the print spooler service, which starts during boot,
    must be restarted in order for them to run. The print spooler service runs under
    SYSTEM level permissions, therefore print processors installed by an adversary
    may run under elevated privileges.'
- source_sentence: Heyoka Backdoor has been spread through malicious document lures.
  sentences:
  - 'T1027.015: Adversaries may use compression to obfuscate their payloads or files.
    Compressed file formats such as ZIP, gzip, 7z, and RAR can compress and archive
    multiple files together to make it easier and faster to transfer files. In addition
    to compressing files, adversaries may also compress shellcode directly - for example,
    in order to store it in a Windows Registry key (i.e., [Fileless Storage](https://attack.mitre.org/techniques/T1027/011)).
    In order to further evade detection, adversaries may combine multiple ZIP files
    into one archive. This process of concatenation creates an archive that appears
    to be a single archive but in fact contains the central directories of the embedded
    archives. Some ZIP readers, such as 7zip, may not be able to identify concatenated
    ZIP files and miss the presence of the malicious payload. File archives may be
    sent as one [Spearphishing Attachment](https://attack.mitre.org/techniques/T1566/001)
    through email. Adversaries have sent malicious payloads as archived files to encourage
    the user to interact with and extract the malicious payload onto their system
    (i.e., [Malicious File](https://attack.mitre.org/techniques/T1204/002)). However,
    some file compression tools, such as 7zip, can be used to produce self-extracting
    archives. Adversaries may send self-extracting archives to hide the functionality
    of their payload and launch it without requiring multiple actions from the user.
    [Compression](https://attack.mitre.org/techniques/T1027/015) may be used in combination
    with [Encrypted/Encoded File](https://attack.mitre.org/techniques/T1027/013) where
    compressed files are encrypted and password-protected.'
  - 'T1204.002: An adversary may rely upon a user opening a malicious file in order
    to gain execution. Users may be subjected to social engineering to get them to
    open a file that will lead to code execution. This user action will typically
    be observed as follow-on behavior from [Spearphishing Attachment](https://attack.mitre.org/techniques/T1566/001).
    Adversaries may use several types of files that require a user to execute them,
    including .doc, .pdf, .xls, .rtf, .scr, .exe, .lnk, .pif, .cpl, .reg, and .iso.
    Adversaries may employ various forms of [Masquerading](https://attack.mitre.org/techniques/T1036)
    and [Obfuscated Files or Information](https://attack.mitre.org/techniques/T1027)
    to increase the likelihood that a user will open and successfully execute a malicious
    file. These methods may include using a familiar naming convention and/or password
    protecting the file and supplying instructions to a user on how to open it. While
    [Malicious File](https://attack.mitre.org/techniques/T1204/002) frequently occurs
    shortly after Initial Access it may occur at other phases of an intrusion, such
    as when an adversary places a file in a shared directory or on a user''s desktop
    hoping that a user will click on it. This activity may also be seen shortly after
    [Internal Spearphishing](https://attack.mitre.org/techniques/T1534).'
  - 'T1087.003: Adversaries may attempt to get a listing of email addresses and accounts.
    Adversaries may try to dump Exchange address lists such as global address lists
    (GALs). In on-premises Exchange and Exchange Online, the <code>Get-GlobalAddressList</code>
    PowerShell cmdlet can be used to obtain email addresses and accounts from a domain
    using an authenticated session. In Google Workspace, the GAL is shared with Microsoft
    Outlook users through the Google Workspace Sync for Microsoft Outlook (GWSMO)
    service. Additionally, the Google Workspace Directory allows for users to get
    a listing of other users within the organization.'
---

# SentenceTransformer based on sentence-transformers/all-mpnet-base-v2

This is a [sentence-transformers](https://www.SBERT.net) model finetuned from [sentence-transformers/all-mpnet-base-v2](https://huggingface.co/sentence-transformers/all-mpnet-base-v2). It maps sentences & paragraphs to a 768-dimensional dense vector space and can be used for semantic textual similarity, semantic search, paraphrase mining, text classification, clustering, and more.

## Model Details

### Model Description
- **Model Type:** Sentence Transformer
- **Base model:** [sentence-transformers/all-mpnet-base-v2](https://huggingface.co/sentence-transformers/all-mpnet-base-v2) <!-- at revision e8c3b32edf5434bc2275fc9bab85f82640a19130 -->
- **Maximum Sequence Length:** 384 tokens
- **Output Dimensionality:** 768 dimensions
- **Similarity Function:** Cosine Similarity
<!-- - **Training Dataset:** Unknown -->
<!-- - **Language:** Unknown -->
<!-- - **License:** Unknown -->

### Model Sources

- **Documentation:** [Sentence Transformers Documentation](https://sbert.net)
- **Repository:** [Sentence Transformers on GitHub](https://github.com/UKPLab/sentence-transformers)
- **Hugging Face:** [Sentence Transformers on Hugging Face](https://huggingface.co/models?library=sentence-transformers)

### Full Model Architecture

```
SentenceTransformer(
  (0): Transformer({'max_seq_length': 384, 'do_lower_case': False}) with Transformer model: MPNetModel 
  (1): Pooling({'word_embedding_dimension': 768, 'pooling_mode_cls_token': False, 'pooling_mode_mean_tokens': True, 'pooling_mode_max_tokens': False, 'pooling_mode_mean_sqrt_len_tokens': False, 'pooling_mode_weightedmean_tokens': False, 'pooling_mode_lasttoken': False, 'include_prompt': True})
  (2): Normalize()
)
```

## Usage

### Direct Usage (Sentence Transformers)

First install the Sentence Transformers library:

```bash
pip install -U sentence-transformers
```

Then you can load this model and run inference.
```python
from sentence_transformers import SentenceTransformer

# Download from the 🤗 Hub
model = SentenceTransformer("sentence_transformers_model_id")
# Run inference
sentences = [
    'Heyoka Backdoor has been spread through malicious document lures.',
    "T1204.002: An adversary may rely upon a user opening a malicious file in order to gain execution. Users may be subjected to social engineering to get them to open a file that will lead to code execution. This user action will typically be observed as follow-on behavior from [Spearphishing Attachment](https://attack.mitre.org/techniques/T1566/001). Adversaries may use several types of files that require a user to execute them, including .doc, .pdf, .xls, .rtf, .scr, .exe, .lnk, .pif, .cpl, .reg, and .iso. Adversaries may employ various forms of [Masquerading](https://attack.mitre.org/techniques/T1036) and [Obfuscated Files or Information](https://attack.mitre.org/techniques/T1027) to increase the likelihood that a user will open and successfully execute a malicious file. These methods may include using a familiar naming convention and/or password protecting the file and supplying instructions to a user on how to open it. While [Malicious File](https://attack.mitre.org/techniques/T1204/002) frequently occurs shortly after Initial Access it may occur at other phases of an intrusion, such as when an adversary places a file in a shared directory or on a user's desktop hoping that a user will click on it. This activity may also be seen shortly after [Internal Spearphishing](https://attack.mitre.org/techniques/T1534).",
    'T1087.003: Adversaries may attempt to get a listing of email addresses and accounts. Adversaries may try to dump Exchange address lists such as global address lists (GALs). In on-premises Exchange and Exchange Online, the <code>Get-GlobalAddressList</code> PowerShell cmdlet can be used to obtain email addresses and accounts from a domain using an authenticated session. In Google Workspace, the GAL is shared with Microsoft Outlook users through the Google Workspace Sync for Microsoft Outlook (GWSMO) service. Additionally, the Google Workspace Directory allows for users to get a listing of other users within the organization.',
]
embeddings = model.encode(sentences)
print(embeddings.shape)
# [3, 768]

# Get the similarity scores for the embeddings
similarities = model.similarity(embeddings, embeddings)
print(similarities.shape)
# [3, 3]
```

<!--
### Direct Usage (Transformers)

<details><summary>Click to see the direct usage in Transformers</summary>

</details>
-->

<!--
### Downstream Usage (Sentence Transformers)

You can finetune this model on your own dataset.

<details><summary>Click to expand</summary>

</details>
-->

<!--
### Out-of-Scope Use

*List how the model may foreseeably be misused and address what users ought not to do with the model.*
-->

<!--
## Bias, Risks and Limitations

*What are the known or foreseeable issues stemming from this model? You could also flag here known failure cases or weaknesses of the model.*
-->

<!--
### Recommendations

*What are recommendations with respect to the foreseeable issues? For example, filtering explicit content.*
-->

## Training Details

### Training Dataset

#### Unnamed Dataset


* Size: 10,432 training samples
* Columns: <code>sentence_0</code> and <code>sentence_1</code>
* Approximate statistics based on the first 1000 samples:
  |         | sentence_0                                                                         | sentence_1                                                                          |
  |:--------|:-----------------------------------------------------------------------------------|:------------------------------------------------------------------------------------|
  | type    | string                                                                             | string                                                                              |
  | details | <ul><li>min: 3 tokens</li><li>mean: 43.05 tokens</li><li>max: 384 tokens</li></ul> | <ul><li>min: 73 tokens</li><li>mean: 276.9 tokens</li><li>max: 384 tokens</li></ul> |
* Samples:
  | sentence_0                                                                                                                                                                                                | sentence_1                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               |
  |:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
  | <code>Mustang Panda used LNK files to execute PowerShell commands leading to eventual PlugX installation during RedDelta Modified PlugX Infection Chain Operations.</code>                                | <code>T1059.001: Adversaries may abuse PowerShell commands and scripts for execution. PowerShell is a powerful interactive command-line interface and scripting environment included in the Windows operating system. Adversaries can use PowerShell to perform a number of actions, including discovery of information and execution of code. Examples include the <code>Start-Process</code> cmdlet which can be used to run an executable and the <code>Invoke-Command</code> cmdlet which runs a command locally or on a remote computer (though administrator permissions are required to use PowerShell to connect to remote systems). PowerShell may also be used to download and run executables from the Internet, which can be executed from disk or in memory without touching disk. A number of PowerShell-based offensive testing tools are available, including [Empire](https://attack.mitre.org/software/S0363), [PowerSploit](https://attack.mitre.org/software/S0194), [PoshC2](https://attack.mitre.org/software/S0378), an...</code> |
  | <code>Empire leverages PowerShell for the majority of its client-side agent tasks. Empire also contains the ability to conduct PowerShell remoting with the <code>Invoke-PSRemoting</code> module.</code> | <code>T1059.001: Adversaries may abuse PowerShell commands and scripts for execution. PowerShell is a powerful interactive command-line interface and scripting environment included in the Windows operating system. Adversaries can use PowerShell to perform a number of actions, including discovery of information and execution of code. Examples include the <code>Start-Process</code> cmdlet which can be used to run an executable and the <code>Invoke-Command</code> cmdlet which runs a command locally or on a remote computer (though administrator permissions are required to use PowerShell to connect to remote systems). PowerShell may also be used to download and run executables from the Internet, which can be executed from disk or in memory without touching disk. A number of PowerShell-based offensive testing tools are available, including [Empire](https://attack.mitre.org/software/S0363), [PowerSploit](https://attack.mitre.org/software/S0194), [PoshC2](https://attack.mitre.org/software/S0378), an...</code> |
  | <code>Batch files create a new admin user [T1078.002], force a group policy update, set pertinent registry keys to auto-extract</code>                                                                    | <code>T1484.001: Adversaries may modify Group Policy Objects (GPOs) to subvert the intended discretionary access controls for a domain, usually with the intention of escalating privileges on the domain. Group policy allows for centralized management of user and computer settings in Active Directory (AD). GPOs are containers for group policy settings made up of files stored within a predictable network path `\<DOMAIN>\SYSVOL\<DOMAIN>\Policies\`. Like other objects in AD, GPOs have access controls associated with them. By default all user accounts in the domain have permission to read GPOs. It is possible to delegate GPO access control permissions, e.g. write access, to specific users or groups in the domain. Malicious GPO modifications can be used to implement many other malicious behaviors such as [Scheduled Task/Job](https://attack.mitre.org/techniques/T1053), [Disable or Modify Tools](https://attack.mitre.org/techniques/T1562/001), [Ingress Tool Transfer](https://attack.mitre.org/technique...</code> |
* Loss: [<code>MultipleNegativesRankingLoss</code>](https://sbert.net/docs/package_reference/sentence_transformer/losses.html#multiplenegativesrankingloss) with these parameters:
  ```json
  {
      "scale": 20.0,
      "similarity_fct": "cos_sim"
  }
  ```

### Training Hyperparameters
#### Non-Default Hyperparameters

- `per_device_train_batch_size`: 32
- `per_device_eval_batch_size`: 32
- `num_train_epochs`: 1
- `multi_dataset_batch_sampler`: round_robin

#### All Hyperparameters
<details><summary>Click to expand</summary>

- `overwrite_output_dir`: False
- `do_predict`: False
- `eval_strategy`: no
- `prediction_loss_only`: True
- `per_device_train_batch_size`: 32
- `per_device_eval_batch_size`: 32
- `per_gpu_train_batch_size`: None
- `per_gpu_eval_batch_size`: None
- `gradient_accumulation_steps`: 1
- `eval_accumulation_steps`: None
- `torch_empty_cache_steps`: None
- `learning_rate`: 5e-05
- `weight_decay`: 0.0
- `adam_beta1`: 0.9
- `adam_beta2`: 0.999
- `adam_epsilon`: 1e-08
- `max_grad_norm`: 1
- `num_train_epochs`: 1
- `max_steps`: -1
- `lr_scheduler_type`: linear
- `lr_scheduler_kwargs`: {}
- `warmup_ratio`: 0.0
- `warmup_steps`: 0
- `log_level`: passive
- `log_level_replica`: warning
- `log_on_each_node`: True
- `logging_nan_inf_filter`: True
- `save_safetensors`: True
- `save_on_each_node`: False
- `save_only_model`: False
- `restore_callback_states_from_checkpoint`: False
- `no_cuda`: False
- `use_cpu`: False
- `use_mps_device`: False
- `seed`: 42
- `data_seed`: None
- `jit_mode_eval`: False
- `use_ipex`: False
- `bf16`: False
- `fp16`: False
- `fp16_opt_level`: O1
- `half_precision_backend`: auto
- `bf16_full_eval`: False
- `fp16_full_eval`: False
- `tf32`: None
- `local_rank`: 0
- `ddp_backend`: None
- `tpu_num_cores`: None
- `tpu_metrics_debug`: False
- `debug`: []
- `dataloader_drop_last`: False
- `dataloader_num_workers`: 0
- `dataloader_prefetch_factor`: None
- `past_index`: -1
- `disable_tqdm`: False
- `remove_unused_columns`: True
- `label_names`: None
- `load_best_model_at_end`: False
- `ignore_data_skip`: False
- `fsdp`: []
- `fsdp_min_num_params`: 0
- `fsdp_config`: {'min_num_params': 0, 'xla': False, 'xla_fsdp_v2': False, 'xla_fsdp_grad_ckpt': False}
- `fsdp_transformer_layer_cls_to_wrap`: None
- `accelerator_config`: {'split_batches': False, 'dispatch_batches': None, 'even_batches': True, 'use_seedable_sampler': True, 'non_blocking': False, 'gradient_accumulation_kwargs': None}
- `deepspeed`: None
- `label_smoothing_factor`: 0.0
- `optim`: adamw_torch
- `optim_args`: None
- `adafactor`: False
- `group_by_length`: False
- `length_column_name`: length
- `ddp_find_unused_parameters`: None
- `ddp_bucket_cap_mb`: None
- `ddp_broadcast_buffers`: False
- `dataloader_pin_memory`: True
- `dataloader_persistent_workers`: False
- `skip_memory_metrics`: True
- `use_legacy_prediction_loop`: False
- `push_to_hub`: False
- `resume_from_checkpoint`: None
- `hub_model_id`: None
- `hub_strategy`: every_save
- `hub_private_repo`: False
- `hub_always_push`: False
- `gradient_checkpointing`: False
- `gradient_checkpointing_kwargs`: None
- `include_inputs_for_metrics`: False
- `eval_do_concat_batches`: True
- `fp16_backend`: auto
- `push_to_hub_model_id`: None
- `push_to_hub_organization`: None
- `mp_parameters`: 
- `auto_find_batch_size`: False
- `full_determinism`: False
- `torchdynamo`: None
- `ray_scope`: last
- `ddp_timeout`: 1800
- `torch_compile`: False
- `torch_compile_backend`: None
- `torch_compile_mode`: None
- `dispatch_batches`: None
- `split_batches`: None
- `include_tokens_per_second`: False
- `include_num_input_tokens_seen`: False
- `neftune_noise_alpha`: None
- `optim_target_modules`: None
- `batch_eval_metrics`: False
- `eval_on_start`: False
- `use_liger_kernel`: False
- `eval_use_gather_object`: False
- `prompts`: None
- `batch_sampler`: batch_sampler
- `multi_dataset_batch_sampler`: round_robin

</details>

### Framework Versions
- Python: 3.9.12
- Sentence Transformers: 3.3.0
- Transformers: 4.45.2
- PyTorch: 2.5.1+cu124
- Accelerate: 0.27.0
- Datasets: 2.20.0
- Tokenizers: 0.20.0

## Citation

### BibTeX

#### Sentence Transformers
```bibtex
@inproceedings{reimers-2019-sentence-bert,
    title = "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks",
    author = "Reimers, Nils and Gurevych, Iryna",
    booktitle = "Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing",
    month = "11",
    year = "2019",
    publisher = "Association for Computational Linguistics",
    url = "https://arxiv.org/abs/1908.10084",
}
```

#### MultipleNegativesRankingLoss
```bibtex
@misc{henderson2017efficient,
    title={Efficient Natural Language Response Suggestion for Smart Reply},
    author={Matthew Henderson and Rami Al-Rfou and Brian Strope and Yun-hsuan Sung and Laszlo Lukacs and Ruiqi Guo and Sanjiv Kumar and Balint Miklos and Ray Kurzweil},
    year={2017},
    eprint={1705.00652},
    archivePrefix={arXiv},
    primaryClass={cs.CL}
}
```

<!--
## Glossary

*Clearly define terms in order to be accessible across audiences.*
-->

<!--
## Model Card Authors

*Lists the people who create the model card, providing recognition and accountability for the detailed work that goes into its construction.*
-->

<!--
## Model Card Contact

*Provides a way for people who have updates to the Model Card, suggestions, or questions, to contact the Model Card authors.*
-->