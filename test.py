import dspy
from dspy import OllamaLocal
from dspy import Example
import dspy.evaluate
from dspy.teleprompt import BootstrapFewShot

from embeddings.mixedbread import MixBreadEmbeddings
from vectordb.qdrant import CustomQdrantRetriever, QdrantClient

# Embedding
embedding_model = MixBreadEmbeddings(
    model_name="mixedbread-ai/mxbai-embed-large-v1"
)

# Set up Retriever
qdrant_client = QdrantClient(url="http://localhost:6333")
retriever = CustomQdrantRetriever(
    qdrant_collection_name="tfdocs", 
    qdrant_client=qdrant_client, 
    embedding_model=embedding_model,
    k=5,
)

# Set up LLM
llm = OllamaLocal(
    model="mistral:latest", 
    model_type="chat", 
    max_tokens=1024, 
    top_p=1, 
    top_k=20, 
    base_url="http://localhost:11434", 
)

# Configure the settings
dspy.configure(lm=llm, rm=retriever)

# Generate Signature for Input
class GenerateAnswer(dspy.Signature):
    """Answer questions with detailed factual answers."""

    context = dspy.InputField(desc="Contains relevant facts or necessary context to answer the question")
    question = dspy.InputField()
    answer = dspy.OutputField(desc="Detailed answer with respect to given question and context")

# Create a RAG module
class RAG(dspy.Module):

    def __init__(self, retriever: dspy.Retrieve, k: int = 5):
        super().__init__()
        
        self.k = k
        self.retriever = retriever
        self.generate_answer = dspy.ChainOfThought(signature=GenerateAnswer)

    def forward(self, question, k=None):
        retrieved = self.retriever(question, k)
        prediction = self.generate_answer(context=retrieved.passages, question=question)
        return dspy.Prediction(context=retrieved.passages, answer=prediction.answer, metadata=retrieved.metadata, score=retrieved.probs)
    
# Ask any question you like to this simple RAG program.
my_question = "What is Truefoundry?"
# my_question = ["What is Truefoundry?", "What is service?"]
uncompiled_rag = RAG(retriever=retriever, k=2)
# Get the prediction. This contains `pred.context` and `pred.answer`.
prediction = uncompiled_rag(my_question)

print(f"Answer: {prediction.answer}")
print("====================================")

for context, metadata in zip(prediction.context, prediction.metadata):
    print(f"Context: {context}")
    print("------------------------------------")
    print(f"Metadata: {metadata}")
    print("====================================\n\n")


llm.inspect_history(n=1)




# Generate 10 training and 20 validation examples
train = [
    {
        "question" : "What factors influence the dynamic adjustment of the number of replicas in autoscaling, and how is the autoscaling strategy determined based on the Queue Backlog?",
        "answer" : "The dynamic adjustment of the number of replicas in autoscaling is influenced by the fluctuation in traffic or resource usage. Specifically, the autoscaling strategy is determined based on the Queue Backlog. When traffic or resource usage varies, the autoscaling strategy evaluates the Queue Backlog to decide the appropriate number of replicas between the defined minimum and maximum replica counts."
    }, 
    {
        "question" : "What role do autoscaling metrics play in dynamically adjusting resource allocation for an async service?",
        "answer" : "Autoscaling metrics play a crucial role in dynamically adjusting resource allocation for an async service by monitoring and responding to changing demands while maintaining optimal performance. These metrics, such as AWS SQS Average Backlog, provide insights into the queue length and help the autoscaler determine the appropriate resource allocation to handle incoming requests efficiently."
    },
    {
        "question" : "What is the purpose of creating teams within the Truefoundry platform, and how does it streamline resource management?",
        "answer" : "Creating teams within the Truefoundry platform serves to streamline resource management by simplifying access control and allocation processes. Teams allow users to group individuals with similar responsibilities or access requirements, reducing the need to individually assign permissions for each resource. Additionally, teams facilitate efficient collaboration by providing a structured framework for managing access permissions across multiple resources."
    },
    {
        "question" : "How does the Truefoundry platform support role-based access control (RBAC) for managing user permissions?",
        "answer" : "The Truefoundry platform supports role-based access control (RBAC) by assigning specific roles to users based on their responsibilities and access requirements. Each role defines a set of permissions that determine the actions users can perform on resources within the platform. By assigning roles to users, administrators can enforce security policies, restrict unauthorized access, and ensure compliance with data protection regulations."
    },
    {
        "question" : "What are the key benefits of using the Truefoundry platform for managing cloud resources?",
        "answer" : "The Truefoundry platform offers several key benefits for managing cloud resources, including centralized resource management, automated provisioning, and enhanced security features. By providing a unified interface to monitor and control cloud resources, Truefoundry simplifies resource allocation, reduces operational overhead, and improves scalability. Additionally, automated provisioning capabilities streamline resource deployment processes, while security features such as RBAC and encryption enhance data protection and compliance."
    },
    {
        "question" : "What are the steps involved in creating your account on TrueFoundry?",
        "answer" : "Navigate to the `create your account` page on the TrueFoundry website. Fill out the registration form with your company name, work email, username, and password. Click the `Create Account` button to submit the form."
    },
    {
        "question" : "What cloud providers are supported for creating Kubernetes clusters on TrueFoundry, and what are the recommended options for accessing all platform features?",
        "answer" : "TrueFoundry supports AWS EKS, GCP GKE, and Azure AKS for creating Kubernetes clusters. For accessing all platform features, it is recommended to use one of these major cloud providers. Note that kind and minikube, while supported for local clusters, may not support all platform features."
    },
    {
        "question" : "What are TrueFoundry jobs, and in what scenarios are they particularly well-suited?",
        "answer" : "TrueFoundry jobs are task-oriented workloads designed to run for a specific duration to complete a task and then terminate, releasing the resources. They are well-suited for scenarios such as model training on large datasets, routine maintenance tasks like data backups and report generation, and large-scale batch inference tasks."
    },
    {
        "question" : "What is an MLRepo in Truefoundry, and how does it differ from Git repositories?",
        "answer" : "An MLRepo in Truefoundry serves the purpose of versioning ML models, artifacts, and metadata, similar to how Git repositories version code. However, MLRepos are specifically tailored for ML assets, and access to them can be granted to workspaces, enabling secure and controlled access to ML assets across teams and applications."
    }
]


test = [
    {
        "question" : "How can you view the details of a job run in TrueFoundry?",
        "answer" : "You can view the details of a job run in TrueFoundry by accessing the Job Run section. This section provides information about the status and progress of the job run"
    },
    {
        "question" : "What are key design principles of truefoundry?",
        "answer" : "The key design principles of TrueFoundry are: Cloud Native: TrueFoundry operates on Kubernetes, allowing it to function on various cloud providers or on-premises environments. ML Inherits the same SRE principles as the rest of the infrastructure: TrueFoundry seamlessly integrates with your existing software stack, providing ML teams with the same SRE (Site Reliability Engineering), security, and cost optimization features. No Vendor Lockin: TrueFoundry is designed to avoid vendor lock-in. It ensures easy migration by providing accessible APIs and exposing all Kubernetes manifests generated, enabling smooth transition if needed."
    },
    {
        "question" : "What architecture does TrueFoundry follow, and what benefits does it offer?",
        "answer" : "TrueFoundry follows a split-plane architecture, enabling both on-premises deployment and ensuring that service reliability does not rely solely on TrueFoundry. This architecture enhances reliability and flexibility while allowing for customization based on specific organizational needs."
    },
    {
        "question" : "How is the organization of workspaces typically structured within a cluster?",
        "answer" : "Workspaces within a cluster can be organized based on teams, applications, and environments. For example, different teams may manage various applications, each with its own set of environments such as development, staging, and production."
    },
    {
        "question" : "How can a user create a workspace in Truefoundry?",
        "answer" : "To create a workspace in Truefoundry, users can navigate to the Workspace tab in the platform and click on the `Create Workspace` button. Once created, users can obtain the Fully Qualified Name (FQN) of the workspace from the FQN button."
    },
    {
        "question" : "What is the process for creating an ML Repo in Truefoundry?",
        "answer" : "To create an ML Repo in Truefoundry, users need to have at least one Storage Integration configured. They can then access the list of storage integrations from the dropdown menu and select one to associate with the ML Repo. After selecting a storage integration, users can create an ML Repo from the ML Repo's tab in the platform."
    },
]


# trainset = [Example(**data) for data in train]
# testset = [Example(**data) for data in test]

# # Tell DSPy that the 'question' field is the input. Any other fields are labels and/or metadata.
# trainset = [x.with_inputs('question') for x in trainset]
# devset = [x.with_inputs('question') for x in testset]



# # Validation logic: check that the predicted answer is correct.
# # Also check that the retrieved context does actually contain that answer.
# def validate_context_and_answer(example, pred, trace=None):
#     answer_EM = dspy.evaluate.answer_exact_match(example, pred)
#     answer_PM = dspy.evaluate.answer_passage_match(example, pred)
#     return answer_EM and answer_PM

# # Set up a basic optimizer, which will compile our RAG program.
# optimizer = BootstrapFewShot(metric=validate_context_and_answer)

# # Compile!
# compiled_rag = optimizer.compile(RAG(retriever=retriever, k=5), trainset=trainset)