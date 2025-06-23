import streamlit as st
from py2neo import Graph
from langchain_community.llms import Ollama

import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, date
import pandas as pd
# Connect to the Neo4j databases

from dotenv import load_dotenv
import os

# Load variables from .env file
load_dotenv()

# Get credentials from environment
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USER = os.getenv("NEO4J_USER")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

user_graph = Graph(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD), name="interactions")
pmbok_graph = Graph(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD), name="propertykg")
reddit_graph = Graph(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD), name="reddit")

# Initialize session state to track project session start and page navigation
if 'project_started' not in st.session_state:
    st.session_state['project_started'] = False
if 'page' not in st.session_state:
    st.session_state['page'] = 'Dashboard'


llama_model = Ollama(model="llama3.1", temperature=0.3)

def get_existing_projects(user_id):
    return user_graph.run("""
        MATCH (u:User {id: $user_id})-[:TRACKS]->(p:Project)
        RETURN p.name AS ProjectName
    """, user_id=user_id).data()

def get_project_details(user_id, project_name):
    return user_graph.run("""
        MATCH (u:User {id: $user_id})-[:TRACKS]->(p:Project {name: $project_name}) 
        RETURN p.description AS Description, p.budget AS Budget, p.scope AS Scope
    """, user_id=user_id, project_name=project_name).data()

def update_project(user_id, project_name, updated_data):
    user_graph.run("""
        MATCH (u:User {id: $user_id})-[:TRACKS]->(p:Project {name: $project_name})
        SET p.description = $description, 
            p.budget = $budget, 
            p.scope = $scope,
            p.status = $status  // Add status field here
    """, user_id=user_id, project_name=project_name, **updated_data)


def create_milestone(user_id, project_name, milestone_data):
    # Query to create a new milestone and associate it with the project
    query = """
        MATCH (u:User {id: $user_id})-[:TRACKS]->(p:Project {name: $project_name})
        CREATE (m:Milestone {name: $milestone_name, description: $milestone_description, deadline: $milestone_deadline, status: $milestone_status})
        MERGE (p)-[:HAS_MILESTONE]->(m)
    """
    user_graph.run(query, 
                   user_id=user_id, 
                   project_name=project_name,
                   milestone_name=milestone_data['milestone_name'],
                   milestone_description=milestone_data['milestone_description'],
                   milestone_deadline=milestone_data['milestone_deadline'],
                   milestone_status=milestone_data['milestone_status'])

def get_milestone_details(user_id, project_name, milestone_name):
    try:
        query = """
            MATCH (u:User {id: $user_id})-[:TRACKS]->(p:Project {name: $project_name})-[:HAS_MILESTONE]->(m:Milestone {name: $milestone_name})
            RETURN m.name AS Name, m.description AS Description, m.deadline AS Deadline, m.status AS Status
        """
        result = user_graph.run(query, user_id=user_id, project_name=project_name, milestone_name=milestone_name).data()
        return result[0] if result else {}
    except Exception as e:
        print(f"Error fetching milestone details: {e}")
        return {}
    
    
def get_milestone_names(user_id, project_name):
    # Query to fetch milestone names for the selected project
    query = """
        MATCH (u:User {id: $user_id})-[:TRACKS]->(p:Project {name: $project_name})-[:HAS_MILESTONE]->(m:Milestone)
        RETURN m.name AS MilestoneName
    """
    result = user_graph.run(query, user_id=user_id, project_name=project_name).data()
    
    # Return list of milestone names
    return [milestone['MilestoneName'] for milestone in result]
def update_milestone(user_id, project_name, milestone_name, milestone_data):
    # Query to update an existing milestone
    query = """
        MATCH (u:User {id: $user_id})-[:TRACKS]->(p:Project {name: $project_name})-[:HAS_MILESTONE]->(m:Milestone {name: $milestone_name})
        SET m.name = $milestone_name, m.description = $milestone_description, m.deadline = $milestone_deadline, m.status = $milestone_status
    """
    user_graph.run(query, 
                   user_id=user_id, 
                   project_name=project_name,
                   milestone_name=milestone_name,
                   milestone_description=milestone_data['milestone_description'],
                   milestone_deadline=milestone_data['milestone_deadline'],
                   milestone_status=milestone_data['milestone_status'])

import streamlit as st
from neo4j import GraphDatabase

# Assuming user_graph is your Neo4j GraphDatabase connection

import threading
import streamlit as st
from neo4j import GraphDatabase

# Assuming user_graph is your Neo4j GraphDatabase connection

import streamlit as st
from neo4j import GraphDatabase



def update_task(user_id, project_name, task_name, updated_data):
    """Update task details in the Neo4j database."""
    user_graph.run(
        """
        MATCH (u:User {id: $user_id})-[:TRACKS]->(p:Project {name: $project_name})-[:INCLUDES]->(t:Task {name: $task_name})
        SET t.status = $status, t.due_date = $due_date, t.start_date = $start_date, t.description = $description, t.assigned_to = $assigned_to
        """,
        user_id=user_id, 
        project_name=project_name,
        task_name=task_name,
        status=updated_data['status'],
        due_date=updated_data['due_date'],
        start_date=updated_data['start_date'],
        description=updated_data['description'],
        assigned_to=updated_data['assigned_to']
    )


def update_task(user_id, project_name, task_name, updated_data):
    """Update task details in the Neo4j database."""
    user_graph.run(
        """
        MATCH (u:User {id: $user_id})-[:TRACKS]->(p:Project {name: $project_name})-[:INCLUDES]->(t:Task {name: $task_name})
        SET t.status = $status, t.due_date = $due_date, t.start_date = $start_date, t.description = $description, t.assigned_to = $assigned_to
        """,
        user_id=user_id, 
        project_name=project_name,
        task_name=task_name,
        status=updated_data['status'],
        due_date=updated_data['due_date'],
        start_date=updated_data['start_date'],
        description=updated_data['description'],
        assigned_to=updated_data['assigned_to']
    )


def update_task_status(user_id, project_name, task_name, new_status):
    """Update the status of a task in the Neo4j database."""
    user_graph.run(
        "MATCH (u:User {id: $user_id})-[:TRACKS]->(p:Project {name: $project_name})-[:INCLUDES]->(t:Task {name: $task_name}) "
        "SET t.status = $new_status",
        user_id=user_id, project_name=project_name,
        task_name=task_name, new_status=new_status
    )

def get_task_details_for_gantt(user_id, project_name):
    """Get task details for the Gantt chart."""
    result = user_graph.run("""
        MATCH (u:User {id: $user_id})-[:TRACKS]->(p:Project {name: $project_name})-[:INCLUDES]->(t:Task)
        RETURN t.name AS TaskName, t.start_date AS StartDate, t.due_date AS DueDate, t.status AS Status, t.assigned_to AS AssignedTo
    """, user_id=user_id, project_name=project_name).data()

    # Return the result as a list of task details
    return result

def add_team_member(user_id, project_name, member_name):
    """Add a team member to a project in the Neo4j database."""
    user_graph.run(
        "MATCH (u:User {id: $user_id})-[:TRACKS]->(p:Project {name: $project_name}) "
        "MERGE (m:Member {name: $member_name}) "
        "MERGE (p)-[:HAS_MEMBER]->(m)",
        user_id=user_id, project_name=project_name,
        member_name=member_name
    )
def get_team_members(user_id, project_name):
    return user_graph.run(
        "MATCH (u:User {id: $user_id})-[:TRACKS]->(p:Project {name: $project_name})-[:HAS_MEMBER]->(m:Member) "
        "RETURN m.name AS MemberName",
        user_id=user_id, project_name=project_name
    ).data()



def get_user_progress(user_id, selected_project):
    """Fetch the user progress details for a specific project from the graph database."""
    
    query = """
        MATCH (u:User {id: $user_id})-[:TRACKS]->(p:Project {name: $selected_project})-[:HAS_MILESTONE]->(m:Milestone)-[:INCLUDES]->(t:Task)
        OPTIONAL MATCH (t)-[:HAS_PRIORITY]->(tp:Priority)  // In case there's a priority relationship
        OPTIONAL MATCH (p)-[:HAS_DEADLINE]->(p_deadline:Deadline)
        RETURN p.name AS ProjectName, 
               p.budget AS Budget,
               p.deadline AS ProjectDeadline,
               p.description AS ProjectDescription,
               p.progress AS ProjectProgress,
               p.status AS ProjectStatus,
               m.name AS MilestoneName,
               m.deadline AS MilestoneDeadline,
               t.name AS TaskName, 
               t.due_date AS DueDate, 
               t.status AS TaskStatus,
               t.start_date AS StartDate,
               t.assigned_to AS AssignedTo,
               t.description AS TaskDescription,
               COALESCE(tp.name, 'No Priority') AS TaskPriority
    """
    
    # Execute the query and fetch results
    results = user_graph.run(query, user_id=user_id, selected_project=selected_project).data()

    # If no results are found, return an empty list
    if not results:
        st.warning(f"No progress data found for user {user_id} and project '{selected_project}'")
        return []
    
    # Process results and handle missing or incomplete data
    progress_data = []
    for result in results:
        # Extract project details
        project_name = result.get('ProjectName', 'Unknown Project')
        budget = result.get('Budget', 'No Budget')
        project_deadline = result.get('ProjectDeadline', 'No Deadline')
        project_description = result.get('ProjectDescription', 'No Description')
        project_progress = result.get('ProjectProgress', 'No Progress')
        project_status = result.get('ProjectStatus', 'No Status')

        # Extract milestone details
        milestone_name = result.get('MilestoneName', 'Unknown Milestone')
        milestone_deadline = result.get('MilestoneDeadline', 'No Deadline')

        # Extract task details
        task_name = result.get('TaskName', 'Unnamed Task')
        due_date = result.get('DueDate', 'No Due Date')
        task_status = result.get('TaskStatus', 'No Status')  # Default to 'No Status' if not found
        start_date = result.get('StartDate', 'No Start Date')
        assigned_to = result.get('AssignedTo', 'No Assignee')
        task_description = result.get('TaskDescription', 'No Description')
        task_priority = result.get('TaskPriority', 'No Priority')


        # Append the task details along with project and milestone info into the progress_data list
        progress_data.append({
            'ProjectName': project_name,
            'Budget': budget,
            'ProjectDeadline': project_deadline,
            'ProjectDescription': project_description,
            'ProjectProgress': project_progress,
            'ProjectStatus': project_status,
            'MilestoneName': milestone_name,
            'MilestoneDeadline': milestone_deadline,
            'TaskName': task_name,
            'DueDate': due_date,
            'TaskStatus': task_status,
            'StartDate': start_date,
            'AssignedTo': assigned_to,
            'TaskDescription': task_description,
            'TaskPriority': task_priority
        })
    
    return progress_data

def get_total_tasks(user_id, project_name):
    """Get the total number of tasks for a project."""
    result = user_graph.run("""
        MATCH (u:User {id: $user_id})-[:TRACKS]->(p:Project {name: $project_name})-[:INCLUDES]->(t:Task)
        RETURN COUNT(t) AS total_tasks
    """, user_id=user_id, project_name=project_name).data()
    
    # If the result exists, return the total number of tasks, otherwise return 0
    return result[0]["total_tasks"] if result else 0

# Function to get total number of team members for a project
def get_total_team_members(user_id, project_name):
    """Get the total number of team members for a project."""
    result = user_graph.run("""
        MATCH (u:User {id: $user_id})-[:TRACKS]->(p:Project {name: $project_name})-[:HAS_MEMBER]->(m:Member)
        RETURN COUNT(m) AS total_members
    """, user_id=user_id, project_name=project_name).data()
    
    return result[0]["total_members"] if result else 0
def calculate_kpis(user_id, project_name, user_progress):
    """Calculate key performance indicators based on task completion data."""
    
    # Debugging: Check the user progress data
    print("User Progress Data:", user_progress)

    # Get total tasks using the corrected method
    total_tasks = get_total_tasks(user_id, project_name)
    
    # Calculate completed tasks based on user progress data
    completed_tasks = sum(1 for progress in user_progress if progress.get('TaskStatus') == "Completed")
    
    # Debugging: Check the number of completed tasks
    print(f"Completed Tasks: {completed_tasks}, Total Tasks: {total_tasks}")

    # Calculate the completion rate, ensuring we handle division by zero properly
    kpi_completion_rate = (completed_tasks / total_tasks * 100) if total_tasks > 0 else 0
    
    return {
        "Total Tasks": total_tasks,
        "Completed Tasks": completed_tasks,
        "Completion Rate (%)": kpi_completion_rate
    }


def analyze_deviations(user_progress):
    deviations = []
    current_date = datetime.now().date()
    for progress in user_progress:
        due_date_str = progress.get('DueDate')
        if due_date_str is None:
            deviations.append(f"Task '{progress['TaskName']}' does not have a due date assigned.")
        else:
            try:
                due_date = datetime.strptime(due_date_str, "%Y-%m-%d").date()
                if due_date < current_date:  
                    deviations.append(f"Task '{progress['TaskName']}' is overdue (Due Date: {due_date_str}).")
            except ValueError:
                deviations.append(f"Task '{progress['TaskName']}' has an invalid due date format: {due_date_str}.")
    return deviations

def enhanced_retrieval_pmbok():
    """Retrieve enhanced PMBOK-based insights for project management best practices."""
    try:
        pmbok_context = pmbok_graph.run("""
            MATCH (e:Entity)-[r]->(related:Entity) 
            RETURN DISTINCT e.name AS Name, e.synonyms AS Synonyms, type(r) AS RelationshipType, r.type AS RelationshipProperty
            LIMIT 50
        """).data()
        
        if not pmbok_context:
            print("No PMBOK insights found.")
        
        return pmbok_context
    except Exception as e:
        print(f"Error retrieving PMBOK insights: {e}")
        return []
    
def enhanced_retrieval_reddit():
    """Retrieve filtered Reddit discussions around project management challenges."""
    try:
        reddit_context = reddit_graph.run("""
            MATCH (t:Entity)-[r]->(c:Entity) 
            RETURN t.name AS Title, c.name AS Comment, type(r) AS RelationshipType
            LIMIT 50
        """).data()
        
        if not reddit_context:
            print("No Reddit discussions found.")
        
        return reddit_context
    except Exception as e:
        print(f"Error retrieving Reddit insights: {e}")
        return []
    

def get_projects(user_id):
    """Fetch all projects the user is tracking."""
    query = """
        MATCH (u:User {id: $user_id})-[:TRACKS]->(p:Project)
        RETURN p.name AS ProjectName
    """
    
    # Run the query to fetch the user's projects
    result = user_graph.run(query, user_id=user_id).data()
    
    # If there are projects, return the project names; otherwise, return an empty list
    if result:
        return [project['ProjectName'] for project in result]
    else:
        return []


def answer_progress_and_recommendations(user_id, selected_project):
    """Generate a response summarizing user progress and recommendations based on knowledge graphs."""
    # Get user progress for the selected project
    user_progress = get_user_progress(user_id, selected_project)
    
    if not user_progress:
        return "No project data available for this user."

    progress_summary = "\n".join([  
        f"Project: {progress['ProjectName']}, Milestone: {progress['MilestoneName']}, "
        f"Task: {progress['TaskName']}, Due Date: {progress['DueDate']}, "
        f"Status: {progress.get('Status', 'No Status')}"  # Use .get() to avoid KeyError
        for progress in user_progress
    ])
        
    # Calculate KPIs for the selected project
    kpis = calculate_kpis(user_id, selected_project, user_progress)  # Corrected this line
    
    # Analyze deviations and missed deadlines for the selected project
    deviations = analyze_deviations(user_progress)

    # Retrieve enhanced PMBOK information
    pmbok_context = enhanced_retrieval_pmbok()
    
    if not pmbok_context:
        pmbok_info = "No relevant PMBOK information available."
        pmbok_embeddings = []
    else:
        # Flatten the PMBOK context for better readability in output
        pmbok_info = "\n".join([ 
            f"PMBOK Entity: {result['Name']}, Synonyms: {', '.join(result.get('Synonyms', []))}, Relationship Type: {result.get('RelationshipType', 'None')}, Property Type: {result.get('RelationshipProperty', 'None')}" 
            for result in pmbok_context
        ])
        pmbok_embeddings = [result['Name'] for result in pmbok_context]

    # Retrieve enhanced Reddit information
    reddit_context = enhanced_retrieval_reddit()
    
    if not reddit_context:
        reddit_info = "No relevant Reddit discussions available."
        reddit_embeddings = []
    else:
        reddit_info = "\n".join([ 
            f"Discussion Title: {result['Title']}, Comment: {result['Comment']}, Relationship Type: {result.get('RelationshipType', 'None')}" 
            for result in reddit_context
        ])
        reddit_embeddings = [result['Comment'] for result in reddit_context]

    # Combine all outputs into a structured response
    combined_output = (
        f"### Current Project Status:\n{progress_summary}\n\n"
        f"### Key Performance Indicators (KPIs):\n"
        f"Total Tasks: {kpis['Total Tasks']}\n"
        f"Completed Tasks: {kpis['Completed Tasks']}\n"
        f"Completion Rate: {kpis['Completion Rate (%)']}%\n\n"
        f"### Deviations and Missed Deadlines:\n"
        + "\n".join(deviations) + "\n\n" if deviations else "No deviations detected.\n\n"
        
        f"### PMBOK Knowledge Graph Insights:\n{pmbok_info}\n\n"
        f"**Referenced PMBOK Entities**: {', '.join(pmbok_embeddings) if pmbok_embeddings else 'None'}\n\n"

        f"### Reddit Community Insights:\n{reddit_info}\n\n"
        f"**Referenced Reddit Discussions**: {', '.join(reddit_embeddings) if reddit_embeddings else 'None'}\n\n"
        
        f"### Recommendations Based on PMBOK Insights:\n"
    )

    # Incorporate PMBOK insights into dynamic recommendations and ensure sources are referenced
    for embed in pmbok_embeddings:
        combined_output += (
            f"- **Implement practices related to {embed}**: Use methodologies from {embed} to enhance project efficiency and effectiveness.\n"
        )

    if deviations:
        combined_output += (
            "- **Address overdue tasks**: Focus on resolving overdue tasks promptly to maintain project timelines.\n"
        )

    combined_output += (
        f"### Insights from Reddit Discussions:\n"
        "- **Seek Feedback**: Engage team members for their input on task assignments and expectations, especially for overdue tasks.\n"
    )

    # Generate response using the LLM with additional parameters
    response = llama_model.generate(
        [combined_output],
        max_tokens=500,
        temperature=0.3,
        top_k=50,
        top_p=0.9,
        frequency_penalty=0.5,
        presence_penalty=0.5
    )
    
    text_output = response.generations[0][0].text 

    return text_output


from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px

import plotly.express as px

from datetime import datetime

def analyze_deviations(user_progress):
    """Analyze deviations in the project tasks based on their due dates."""
    deviations = []
    current_date = datetime.now().date()

    for task in user_progress:
        task_name = task.get('TaskName', 'Unknown')
        due_date_str = task.get('DueDate')

        if due_date_str is None:
            deviations.append(f"Task '{task_name}' does not have a due date assigned.")
            continue

        try:
            # Assuming due_date_str is in the format "YYYY-MM-DD"
            due_date = datetime.strptime(str(due_date_str), "%Y-%m-%d").date()

            if due_date < current_date:
                deviations.append(f"Task '{task_name}' is overdue (Due Date: {due_date_str}).")
        except (ValueError, TypeError) as e:
            deviations.append(f"Task '{task_name}' has an invalid due date format: {due_date_str}.")
            print(f"Date parsing error: {e}")
            continue

    return deviations

import plotly.graph_objects as go

def create_progress_chart(user_id, project_name):
    """Generate a progress chart based on the completion rate of tasks."""
    # Fetch task completion data from Neo4j
    query = """
        MATCH (u:User {id: $user_id})-[:TRACKS]->(p:Project {name: $project_name})-[:INCLUDES]->(t:Task)
        RETURN COUNT(t) AS total_tasks, 
               SUM(CASE WHEN t.Status = 'Completed' THEN 1 ELSE 0 END) AS completed_tasks
    """
    task_data = user_graph.run(query, user_id=user_id, project_name=project_name).data()

    # Debugging: Check what data is returned from Neo4j
    print(f"Task Data: {task_data}")  # This will print the raw query result
    
    if not task_data or 'total_tasks' not in task_data[0] or 'completed_tasks' not in task_data[0]:
        print("No valid data found for task completion.")
        return go.Figure()  # Return an empty chart if no data is found or the fields are missing

    total_tasks = task_data[0]['total_tasks']
    completed_tasks = task_data[0]['completed_tasks']

    # Debugging: Check values of total tasks and completed tasks
    print(f"Total Tasks: {total_tasks}, Completed Tasks: {completed_tasks}")

    # Calculate completion rate (ensure no division by zero)
    completion_rate = (completed_tasks / total_tasks) * 100 if total_tasks > 0 else 0

    # Create the progress chart (gauge)
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=completion_rate,
        title={'text': "Project Completion"},
        gauge={
            'axis': {'range': [None, 100]},
            'bar': {'color': "#0073e6"},
            'steps': [
                {'range': [0, 33], 'color': "#ffcccc"},
                {'range': [33, 66], 'color': "#ffedcc"},
                {'range': [66, 100], 'color': "#ccffcc"}
            ],
        }
    ))

    fig.update_layout(height=250, margin=dict(t=20, b=20, l=10, r=10))
    return fig

import plotly.express as px

import plotly.express as px

def create_task_distribution(user_id, project_name):
    """Generate a task distribution chart based on task status."""
    # Fetch task status distribution from Neo4j
    task_status_data = user_graph.run("""
        MATCH (u:User {id: $user_id})-[:TRACKS]->(p:Project {name: $project_name})-[:INCLUDES]->(t:Task)
        RETURN t.Status AS Status, COUNT(t) AS Count
    """, user_id=user_id, project_name=project_name).data()

    # Check if task_status_data is empty
    if not task_status_data:
        print("No tasks found for the specified user and project.")
        return None  # Or handle appropriately
    
    # Initialize a dictionary to count task statuses
    status_counts = {}

    # Process each task status data
    for task in task_status_data:
        status = task.get('Status', 'Unknown')  # Default to 'Unknown' if Status is not available
        count = task.get('Count', 0)
        
        # Increment the count for the specific status
        status_counts[status] = status_counts.get(status, 0) + count

    # Handle cases with no tasks to avoid errors
    if not status_counts:
        status_counts = {"No Tasks": 1}

    # Create a pie chart for task distribution
    fig = px.pie(
        values=list(status_counts.values()),
        names=list(status_counts.keys()),
        title="Task Distribution",
        color_discrete_sequence=px.colors.sequential.Agsunset
    )
    fig.update_traces(textposition='inside', textinfo='percent+label')
    fig.update_layout(height=300, margin=dict(t=20, b=20, l=10, r=10))

    return fig


def capture_user_input(user_id):
    """Capture user input for projects and store it in the Neo4j database."""
    st.markdown("### 📝 Capture Project Details")
    
    # Gather project details
    user_project = st.text_input("Project Name", "Enter project name")
    user_project_description = st.text_area("Project Description", "Enter a brief description of the project")
    user_budget = st.number_input("Project Budget ($)", min_value=0.0, step=100.0)
    user_scope = st.text_area("Project Scope", "Enter the project scope (e.g., deliverables, objectives)")
    project_status = st.selectbox("Project Status", ["Not Started", "In Progress", "Completed", "On Hold", "Cancelled"])

    # Milestones and tasks
    user_milestone = st.text_input("Milestone Name", "Enter milestone name")
    user_milestone_description = st.text_area("Milestone Description", "Enter a brief description of the milestone")
    milestone_deadline = st.date_input("Milestone Deadline", datetime.now()).strftime("%Y-%m-%d")

    user_task = st.text_input("Task Name", "Enter task name")
    user_task_description = st.text_area("Task Description", "Enter a brief description of the task")
    task_due_date = st.date_input("Task Due Date", datetime.now()).strftime("%Y-%m-%d")
    task_status = st.selectbox("Task Status", ["Not Started", "In Progress", "Completed", "Blocked"])

    # Additional elements
    stakeholder_name = st.text_input("Stakeholder Name", "Enter stakeholder name")
    stakeholder_role = st.text_input("Stakeholder Role", "Enter stakeholder role")
    risk_description = st.text_area("Risk Description", "Enter a risk description")
    risk_mitigation_strategy = st.text_area("Risk Mitigation Strategy", "Enter risk mitigation strategy")
    resource_name = st.text_input("Resource Name", "Enter resource name (e.g., team member)")
    resource_role = st.text_input("Resource Role", "Enter resource role")
    sprint_number = st.number_input("Sprint Number", min_value=1, step=1)

    if st.button("Save Project Details"):
        try:
            # Store project information in Neo4j with relationships
            user_graph.run("MERGE (u:User {id: $user_id})", user_id=user_id)
            user_graph.run("""
                MERGE (p:Project {name: $user_project})
                ON CREATE SET p.description = $user_project_description, 
                              p.budget = $user_budget, 
                              p.scope = $user_scope,
                              p.status = $project_status
                ON MATCH SET p.description = $user_project_description, 
                             p.budget = $user_budget, 
                             p.scope = $user_scope,
                             p.status = $project_status
                MERGE (m:Milestone {name: $user_milestone})
                ON CREATE SET m.description = $user_milestone_description,
                              m.deadline = $milestone_deadline
                ON MATCH SET m.description = $user_milestone_description,
                             m.deadline = $milestone_deadline
                MERGE (t:Task {name: $user_task})
                ON CREATE SET t.description = $user_task_description, 
                              t.due_date = $task_due_date, 
                              t.status = $task_status
                ON MATCH SET t.description = $user_task_description, 
                             t.due_date = $task_due_date, 
                             t.status = $task_status
                MERGE (s:Stakeholder {name: $stakeholder_name, role: $stakeholder_role})
                MERGE (r:Risk {description: $risk_description, mitigation_strategy: $risk_mitigation_strategy})
                MERGE (res:Resource {name: $resource_name, role: $resource_role})
                MERGE (spr:Sprint {number: $sprint_number})
            """, 
            user_project=user_project,
            user_project_description=user_project_description,
            user_budget=user_budget,
            user_scope=user_scope,
            project_status=project_status,
            user_milestone=user_milestone,
            user_milestone_description=user_milestone_description,
            milestone_deadline=milestone_deadline,
            user_task=user_task,
            user_task_description=user_task_description,
            task_due_date=task_due_date,
            task_status=task_status,
            stakeholder_name=stakeholder_name,
            stakeholder_role=stakeholder_role,
            risk_description=risk_description,
            risk_mitigation_strategy=risk_mitigation_strategy,
            resource_name=resource_name,
            resource_role=resource_role,
            sprint_number=sprint_number)

            # Create relationships
            user_graph.run("""
                MATCH (u:User {id: $user_id}), 
                      (p:Project {name: $user_project}), 
                      (m:Milestone {name: $user_milestone}), 
                      (t:Task {name: $user_task}),
                      (s:Stakeholder {name: $stakeholder_name}),
                      (r:Risk {description: $risk_description}),
                      (res:Resource {name: $resource_name}),
                      (spr:Sprint {number: $sprint_number})
                MERGE (u)-[:TRACKS]->(p)
                MERGE (p)-[:HAS_MILESTONE]->(m)
                MERGE (m)-[:INCLUDES]->(t)
                MERGE (p)-[:HAS_STAKEHOLDER]->(s)
                MERGE (p)-[:HAS_RISK]->(r)
                MERGE (p)-[:REQUIRES]->(res)
                MERGE (p)-[:INCLUDES_SPRINT]->(spr)
            """, 
            user_id=user_id, 
            user_project=user_project, 
            user_milestone=user_milestone, 
            user_task=user_task,
            stakeholder_name=stakeholder_name,
            risk_description=risk_description,
            resource_name=resource_name,
            sprint_number=sprint_number)

            st.success("✅ Project details saved successfully, including status updates and deadlines!")
        except Exception as e:
            st.error(f"An error occurred while capturing project details: {e}")

# Enhanced CSS Styling with Fixed Colors
st.markdown("""
    <style>
        /* Set a light background for tab content */
    .stTabs [data-baseweb="tab-panel"] {
        background-color: #f8f9fa;
    }
    
    /* Ensure any white text in reports is dark */
    .stMarkdown {
        color: #2c3e50 !important;
    }
    
    /* Keep dashboard header text white but everything else dark */
    .dashboard-header * {
        color: white !important;
    }
    
    /* Make tab text visible */
    .stTabs [data-baseweb="tab"] {
        color: #2c3e50;
    }
    
    /* Keep selected tab text white */
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        color: white !important;
    }
        /* Root variables for consistent theming */
        :root {
            --primary-color: #2962ff;
            --secondary-color: #0d47a1;
            --background-color: #f8f9fa;
            --card-background: #ffffff;
            --text-primary: #2c3e50;
            --text-secondary: #6c757d;
            --success-color: #4caf50;
            --warning-color: #ff9800;
            --danger-color: #f44336;
        }

        /* General page styling */
        .main {
            background-color: var(--background-color);
            padding: 2rem;
        }

        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
        }

        /* Custom container styling */
        .custom-container {
            background-color: var(--card-background);
            border-radius: 12px;
            padding: 1.5rem;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
            margin-bottom: 1.5rem;
            border: 1px solid rgba(0, 0, 0, 0.05);
        }

        /* Dashboard header */
        .dashboard-header {
            text-align: center;
            padding: 2rem 0;
            background: linear-gradient(135deg, var(--primary-color), var(--secondary-color));
            color: white !important;
            border-radius: 12px;
            margin-bottom: 2rem;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }

        .dashboard-header h1 {
            color: white !important;
            margin-bottom: 0.5rem;
        }

        .dashboard-header p {
            color: white !important;
            opacity: 0.9;
        }

        /* Metric cards */
        .metric-card {
            background: var(--card-background);
            border-radius: 10px;
            padding: 1.25rem;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
            transition: transform 0.3s ease;
            border: 1px solid rgba(0, 0, 0, 0.05);
        }

        .metric-card:hover {
            transform: translateY(-5px);
        }

        .metric-value {
            font-size: 1.8rem;
            font-weight: 700;
            color: var(--primary-color);
            text-align: center;
            margin-bottom: 0.5rem;
        }

        .metric-label {
            font-size: 1rem;
            color: var(--text-secondary);
            text-align: center;
            font-weight: 500;
        }

        /* Section headers */
        .section-header {
            font-size: 1.5rem;
            color: var(--text-primary);
            font-weight: 600;
            margin: 2rem 0 1rem 0;
            padding-bottom: 0.5rem;
            border-bottom: 2px solid var(--primary-color);
            background-color: transparent;
        }

        /* Sidebar styling */
        .css-1d391kg {
            background-color: var(--card-background);
        }

        .sidebar-content {
            padding: 1rem;
            background-color: var(--card-background);
        }

        /* Button styling */
        .stButton > button {
            background: linear-gradient(135deg, var(--primary-color), var(--secondary-color));
            color: white !important;
            border-radius: 8px;
            padding: 0.75rem 1.5rem;
            font-weight: 500;
            border: none;
            transition: all 0.3s ease;
            width: 100%;
        }

        .stButton > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
        }

        /* Status badges */
        .status-badge {
            padding: 0.25rem 0.75rem;
            border-radius: 50px;
            font-size: 0.875rem;
            font-weight: 500;
            display: inline-block;
            margin: 0.25rem;
        }

        .status-success {
            background-color: rgba(76, 175, 80, 0.1);
            color: var(--success-color);
        }

        .status-warning {
            background-color: rgba(255, 152, 0, 0.1);
            color: var(--warning-color);
        }

        .status-danger {
            background-color: rgba(244, 67, 54, 0.1);
            color: var(--danger-color);
        }

        /* Chart containers */
        .chart-container {
            background: var(--card-background);
            border-radius: 12px;
            padding: 1rem;
            margin: 1rem 0;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
        }

        /* Form fields */
        .stTextInput > div > div > input {
            border-radius: 8px;
        }

        .stTextArea > div > div > textarea {
            border-radius: 8px;
        }

        /* Expander styling */
        .streamlit-expanderHeader {
            background-color: var(--card-background);
            border-radius: 8px;
            border: 1px solid rgba(0, 0, 0, 0.05);
            color: var(--text-primary) !important;
        }

        /* Progress bars */
        .stProgress > div > div > div {
            background-color: var(--primary-color);
        }

        /* Tab styling */
        .stTabs [data-baseweb="tab-list"] {
            gap: 1rem;
        }

        .stTabs [data-baseweb="tab"] {
            height: auto;
            padding: 0.5rem 1rem;
            color: var(--text-primary) !important;
            border-radius: 4px;
        }

        .stTabs [aria-selected="true"] {
            background-color: var(--primary-color) !important;
            color: white !important;
        }

        /* Warning message styling */
        .stAlert {
            background-color: var(--card-background);
            color: var(--text-primary);
            border: 1px solid rgba(0, 0, 0, 0.05);
            border-radius: 8px;
        }

        /* Markdown text color fix */
        .css-10trblm {
            color: var(--text-primary) !important;
        }

        .css-183lzff {
            color: var(--text-primary) !important;
        }

        /* Footer styling */
        .footer {
            text-align: center;
            padding: 2rem 0;
            color: var(--text-secondary);
            background-color: transparent;
        }

        /* Selectbox styling */
        .stSelectbox [data-baseweb="select"] {
            background-color: var(--card-background);
            border-radius: 8px;
        }

        /* Make sure all text is visible */
        p, h1, h2, h3, h4, h5, h6, span, label {
            color: var(--text-primary) !important;
        }

        /* Exception for dashboard header text */
        .dashboard-header * {
            color: white !important;
        }
    </style>
""", unsafe_allow_html=True)


from datetime import datetime, timedelta
import uuid
import streamlit as st

# Initialize tasks in session state if not already present
if 'tasks' not in st.session_state:
    st.session_state['tasks'] = []

# Initialize page in session state if not already present
if 'page' not in st.session_state:
    st.session_state['page'] = 'Project'  # Default to the project page

# # Function to add a new task
# def add_task(title, description, priority, due_date):
#     if title:
#         st.session_state['tasks'].append({
#             'id': str(uuid.uuid4()),
#             'title': title,
#             'description': description,
#             'priority': priority,
#             'due_date': due_date,
#             'status': 'To Do'
#         })

def delete_task(user_id, project_name, task_name):
    query = """
        MATCH (u:User {id: $user_id})-[:TRACKS]->(p:Project {name: $project_name})-[:INCLUDES]->(t:Task {name: $task_name})
        DETACH DELETE t
    """
    user_graph.run(query, user_id=user_id, project_name=project_name, task_name=task_name)


# Function to update the task status
def update_status(task_id, new_status):
    for task in st.session_state['tasks']:
        if task['id'] == task_id:
            task['status'] = new_status
            break

# Function to edit a task's details
def edit_task(task_id, title, description, priority, due_date):
    for task in st.session_state['tasks']:
        if task['id'] == task_id:
            task['title'] = title
            task['description'] = description
            task['priority'] = priority
            task['due_date'] = due_date
            break

# Render Kanban board with project selector
def render_kanban():
    st.markdown("""
        <style>
        .kanban-column {
            border-radius: 10px;
            padding: 10px;
            margin: 8px;
            min-height: 500px;
            display: flex;
            flex-direction: column;
        }
        .kanban-header {
            text-align: center;
            font-size: 1.2rem;
            font-weight: bold;
            padding: 10px;
            border-radius: 8px 8px 0 0;
            color: white;
        }
        .tasks-container {
            background-color: #f9f9f9;
            border-radius: 0 0 8px 8px;
            padding: 15px;
            flex-grow: 1;
        }
        .task-card {
            border-radius: 10px;
            padding: 12px;
            margin: 12px 0;
            background-color: white;
            border: 1px solid #e0e0e0;
            transition: transform 0.2s ease-in-out;
            overflow: hidden;
            text-overflow: ellipsis;
        }
        .task-card:hover {
            transform: scale(1.02);
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.15);
        }
        .task-title {
            font-size: 1rem;
            font-weight: bold;
            margin-bottom: 5px;
        }
        .task-desc {
            color: #555;
            font-size: 0.9rem;
            margin-bottom: 8px;
            max-height: 60px;
            overflow: auto;
            text-overflow: ellipsis;
        }
        .button-container {
            display: flex;
            gap: 5px;
            margin-top: 8px;
        }
        .priority-high { border-left: 4px solid #e74c3c; }
        .priority-medium { border-left: 4px solid #f39c12; }
        .priority-low { border-left: 4px solid #2ecc71; }
        </style>
    """, unsafe_allow_html=True)

    st.title("Advanced Kanban Board")

    # Project Selector
    existing_projects = get_existing_projects("user123")  # Use a predefined user ID or remove this as needed
    if not existing_projects:
        st.warning("⚠️ No projects found for this user. Please create a new project to get started.")
        return

    project_names = [project["ProjectName"] for project in existing_projects]
    selected_project = st.selectbox("Select Project", project_names, key="project_selector")

    # Task addition form with compact layout
    st.markdown("### Add Task")
    with st.form("Add Task"):
        col1, col2, col3 = st.columns([3, 1, 2])
        with col1:
            title = st.text_input("Title", placeholder="Enter task title", key="task_title")
        with col2:
            priority = st.selectbox("Priority", ["Low", "Medium", "High"], index=1, key="task_priority")
        with col3:
            due_date = st.date_input("Due Date", datetime.today() + timedelta(days=7), key="task_due_date")

        description = st.text_area("Description", placeholder="Enter task details or notes", key="task_description", height=50)
        
        submitted = st.form_submit_button("Add Task")
        
        if submitted:
            add_task(title, description, priority, due_date)
            st.experimental_rerun()

    # Define Kanban columns
    statuses = ["To Do", "In Progress", "Complete"]
    colors = ["#3498db", "#f39c12", "#2ecc71"]
    columns = st.columns(len(statuses))

    for idx, (status, color) in enumerate(zip(statuses, colors)):
        with columns[idx]:
            st.markdown(f"<div class='kanban-header' style='background-color:{color};'>{status}</div>", unsafe_allow_html=True)
            st.markdown("<div class='tasks-container'>", unsafe_allow_html=True)

            for task in [t for t in st.session_state['tasks'] if t['status'] == status]:
                priority_class = f"priority-{task['priority'].lower()}"
                due_date_color = (
                    "#e74c3c" if task['due_date'] <= datetime.today().date()
                    else "#f39c12" if task['due_date'] <= (datetime.today().date() + timedelta(days=3))
                    else "#2ecc71"
                )
                
                st.markdown(f"""
                    <div class='task-card {priority_class}'>
                        <div class='task-title'>{task['title']}</div>
                        <div class='task-desc'>{task['description']}</div>
                        <div style='color: {due_date_color}; font-size: 0.85rem;'>Due: {task['due_date']}</div>
                    </div>
                """, unsafe_allow_html=True)

                col1, col2, col3, col4 = st.columns([1, 1, 1, 1])
                with col1:
                    if status != "To Do":
                        if st.button("⬅️", key=f"back_{task['id']}", help="Move to previous status"):
                            update_status(task['id'], statuses[statuses.index(status) - 1])
                            st.experimental_rerun()
                with col2:
                    if status != "Complete":
                        if st.button("➡️", key=f"forward_{task['id']}", help="Move to next status"):
                            update_status(task['id'], statuses[statuses.index(status) + 1])
                            st.experimental_rerun()
                with col3:
                    if st.button("🖉", key=f"edit_{task['id']}", help="Edit task details"):
                        with st.form(f"edit_form_{task['id']}"):
                            new_title = st.text_input("Title", value=task['title'], key=f"edit_title_{task['id']}")
                            new_description = st.text_area("Description", value=task['description'], key=f"edit_desc_{task['id']}", height=50)
                            new_priority = st.selectbox("Priority", ["Low", "Medium", "High"], 
                                                      index=["Low", "Medium", "High"].index(task['priority']),
                                                      key=f"edit_priority_{task['id']}")
                            new_due_date = st.date_input("Due Date", task['due_date'], key=f"edit_due_date_{task['id']}")
                            if st.form_submit_button("Update Task", key=f"update_task_{task['id']}"):
                                edit_task(task['id'], new_title, new_description, new_priority, new_due_date)
                                st.experimental_rerun()
                with col4:
                    if st.button("🗑️", key=f"delete_{task['id']}", help="Delete task"):
                        delete_task(task['id'])
                        st.experimental_rerun()

            st.markdown("</div>", unsafe_allow_html=True)



            
# Sidebar with navigation buttons
with st.sidebar:
    from PIL import Image
    # Local image path
    image_path = r"C:\Users\chayma.rhaiem\Downloads\ollama.png"

    # Open the image using PIL
    image = Image.open(image_path)

    # Sidebar content with image
    st.sidebar.markdown('<div class="sidebar-content">', unsafe_allow_html=True)
    st.sidebar.image(image, width=150)
    st.markdown("### 👤 User Profile")
    user_id = st.text_input("Enter User ID", value="user123", help="Enter your unique user identifier")
    st.sidebar.header("📝 Capture Project Details")
    
    # Button to start a new project
    if st.button("Start New Project"):
        st.session_state['project_started'] = True
        st.sidebar.success("Project capture session started.")

    # Navigation buttons for Project and Kanban pages
    if st.button("Project Page"):
        st.session_state['page'] = 'Project'
    
    if st.button("Kanban"):
        st.session_state['page'] = 'Kanban'
    
    st.markdown('</div>', unsafe_allow_html=True)

# Page routing
if st.session_state['page'] == 'Kanban':
    render_kanban()
else:
    # Main dashboard (Project Page) code
    st.markdown(f"""
        <div class="custom-container">
            <h3>📊 Active Dashboard: User {user_id}</h3>
        </div>
    """, unsafe_allow_html=True)

    existing_projects = get_existing_projects(user_id)
    if not existing_projects:
        st.warning("⚠️ No projects found for this user. Please create a new project to get started.")
    else:
        project_names = [project["ProjectName"] for project in existing_projects]
        
        # Project selector with improved styling
        st.markdown('<div class="section-header">📁 Project Selection</div>', unsafe_allow_html=True)
        selected_project = st.selectbox(
            "Choose a project to view details",
            project_names,
            key="project_selector"
        )
if selected_project:
    project_details = get_project_details(user_id, selected_project)
    if project_details:
        details = project_details[0]

        total_tasks = get_total_tasks(user_id, selected_project)
        total_team_members = get_total_team_members(user_id, selected_project)

        task_details = get_task_details_for_gantt(user_id, selected_project)
        task_df = pd.DataFrame(task_details)

        if not task_df.empty:
            fig = px.timeline(
                task_df, 
                x_start="StartDate", 
                x_end="DueDate", 
                y="TaskName", 
                color="Status", 
                title="Project Task Gantt Chart",
                labels={"TaskName": "Task Name", "Status": "Task Status"},
                hover_name="AssignedTo",
                hover_data=["StartDate", "DueDate", "Status", "AssignedTo"]
            )
            fig.update_yaxes(categoryorder="total ascending")
            st.plotly_chart(fig)

        # Project Overview Cards

        # Project Overview Cards
        st.markdown('<div class="section-header">📌 Project Overview</div>', unsafe_allow_html=True)

        # Columns for Project Overview Metrics
        col1, col2, col3, col4 = st.columns(4)

        # Project Name Card
        with col1:
            st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{selected_project}</div>
                    <div class="metric-label">Project Name</div>
                </div>
            """, unsafe_allow_html=True)

        # Total Budget Card
        with col2:
            st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">${details['Budget']:,.2f}</div>
                    <div class="metric-label">Total Budget</div>
                </div>
            """, unsafe_allow_html=True)

        # Total Tasks Card
        with col3:
            st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{total_tasks}</div>
                    <div class="metric-label">Total Tasks</div>
                </div>
            """, unsafe_allow_html=True)

        # Total Team Members Card
        with col4:
            st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{total_team_members}</div>
                    <div class="metric-label">Total Members</div>
                </div>
            """, unsafe_allow_html=True)

        # Project Scope (more space, in a separate row)
        st.markdown(f"""
            <div class="metric-card" style="width: 100%; max-width: 800px; margin: 20px auto; padding: 20px;">
                <div class="metric-value" style="white-space: pre-wrap; word-wrap: break-word;">{details['Scope'][:300]}...</div>
                <div class="metric-label" style="text-align: center;">Project Scope</div>
            </div>
        """, unsafe_allow_html=True)

def get_project_details(user_id, project_name):
    return user_graph.run("""
        MATCH (u:User {id: $user_id})-[:TRACKS]->(p:Project {name: $project_name}) 
        RETURN p.description AS Description, p.budget AS Budget, p.scope AS Scope, p.status AS Status
    """, user_id=user_id, project_name=project_name).data()

def update_project_name(user_id, old_project_name, new_project_name):
    user_graph.run("""
        MATCH (u:User {id: $user_id})-[:TRACKS]->(p:Project {name: $old_project_name})
        SET p.name = $new_project_name
    """, user_id=user_id, old_project_name=old_project_name, new_project_name=new_project_name)

with st.expander("Update Project Details"):
    # Fetch the project details
    project_details = get_project_details(user_id, selected_project)

    if project_details:
        details = project_details[0]  # Assuming only one result, as project names are unique

        # Check if 'Status' key exists in details
        project_status = details.get("Status", "Not Started")  # Default to "Not Started" if Status is not found

        # Display existing project details in the form fields
        # Allow user to modify project name
        project_name_updated = st.text_input("Project Name", selected_project)
        
        # Use a unique key for each text_area to avoid duplicate ID errors
        project_description_updated = st.text_area("Project Description", details["Description"], height=100, key=f"description_{selected_project}")
        project_scope_updated = st.text_area("Project Scope", details["Scope"], height=100, key=f"scope_{selected_project}")
        project_budget_updated = st.number_input("Project Budget", value=details["Budget"], min_value=0.0, step=0.01)
        project_status_updated = st.selectbox(
            "Project Status",
            ["Not Started", "In Progress", "Completed"],
            index=["Not Started", "In Progress", "Completed"].index(project_status)  # Use the fetched status
        )

        if st.button("💾 Save Project Changes"):
            # Prepare updated data dictionary
            updated_data = {
                "description": project_description_updated,
                "budget": project_budget_updated,
                "scope": project_scope_updated,
                "status": project_status_updated
            }

            # Check if the project name was changed
            if project_name_updated != selected_project:
                # Ensure you update project name in the relationships as well
                update_project_name(user_id, selected_project, project_name_updated)

            # Update the project in the database
            update_project(user_id, project_name_updated, updated_data)
            st.success(f"✅ Project '{project_name_updated}' updated successfully!")

    else:
        st.info("Project not found. Please ensure the project name is correct.")


import streamlit as st
from datetime import datetime

# Check if milestone details are stored in session state
if "milestone_details" not in st.session_state:
    st.session_state.milestone_details = {}

# Ensure that milestone details are stored in session state
if "milestone_details" not in st.session_state:
    st.session_state.milestone_details = {}

# Project Details Editor Section
st.markdown('<div class="section-header">✏️ Project Details</div>', unsafe_allow_html=True)
with st.expander("Edit Project Information"):
    col1, col2 = st.columns([2, 1])
    
    with col1:
        updated_description = st.text_area("Project Description", details["Description"], height=150)
        updated_scope = st.text_area("Project Scope", value=details["Scope"], height=100)

        # Project Status
        updated_status = st.selectbox(
            "Project Status", 
            ["Not Started", "In Progress", "Completed", "On Hold", "Cancelled"], 
            index=["Not Started", "In Progress", "Completed", "On Hold", "Cancelled"].index(details.get("Status", "Not Started"))
        )

    with col2:
        updated_budget = st.number_input("Budget ($)", value=float(details["Budget"]), min_value=0.0, format="%.2f")

    # Save Project Information
    if st.button("💾 Save Project Information"):
        # Update Project Details
        update_project(user_id, selected_project, {
            "description": updated_description,
            "budget": updated_budget,
            "scope": updated_scope,
            "status": updated_status
        })
        st.success("✅ Project details updated successfully!")


def fetch_milestone_details(user_id, selected_project, selected_milestone):
    query = """
        MATCH (u:User {id: $user_id})-[:TRACKS]->(p:Project {name: $selected_project})-[:HAS_MILESTONE]->(m:Milestone {name: $selected_milestone})
        OPTIONAL MATCH (m)-[:HAS_TASK]->(t:Task)
        RETURN m.name AS Name, m.description AS Description, m.deadline AS Deadline, m.status AS Status, collect(t.name) AS Tasks
    """
    result = user_graph.run(query, user_id=user_id, selected_project=selected_project, selected_milestone=selected_milestone).data()
    
    if result:
        milestone = result[0]
        # Store milestone details and tasks in session state
        st.session_state.milestone_details = milestone
    else:
        st.session_state.milestone_details = {}

import time

def associate_task_with_milestone(user_id, selected_project, selected_milestone, selected_task):
    """Associates an existing task with a milestone in the graph database."""
    
    query = """
        MATCH (u:User {id: $user_id})-[:TRACKS]->(p:Project {name: $selected_project})-[:HAS_MILESTONE]->(m:Milestone {name: $selected_milestone}),
              (t:Task {name: $selected_task})
        CREATE (m)-[:INCLUDES]->(t)
        RETURN m.name AS MilestoneName, t.name AS TaskName
    """
    
    start_time = time.time()

    with st.spinner("Associating task..."):
        # Try-except block to catch any potential database errors
        try:
            result = user_graph.run(query, user_id=user_id, selected_project=selected_project, 
                                    selected_milestone=selected_milestone, selected_task=selected_task).data()

            if result:
                st.success(f"Task '{selected_task}' successfully associated with milestone '{selected_milestone}'.")
            else:
                st.error("Failed to associate task with milestone.")
        except Exception as e:
            st.error(f"Database error: {e}")

    # Calculate and log the time taken for the operation
    elapsed_time = time.time() - start_time
    st.write(f"Operation completed in {elapsed_time:.2f} seconds.")
    
    # Milestone Details Editor Section
st.markdown('<div class="section-header">📅 Milestone Details</div>', unsafe_allow_html=True)
with st.expander("Edit Milestone Information"):
    col1, col2 = st.columns([2, 1])

    # Milestone Selection
    milestone_names = get_milestone_names(user_id, selected_project)
    milestone_names.insert(0, "New Milestone")  # Option for creating a new milestone
    selected_milestone = st.selectbox("Select Milestone to Edit", milestone_names)

    if selected_milestone != "New Milestone":
        # Fetch Milestone Details if not already available in session state
        if 'milestone_details' not in st.session_state or selected_milestone not in st.session_state.milestone_details:
            with st.spinner("Loading milestone details..."):
                try:
                    fetch_milestone_details(user_id, selected_project, selected_milestone)
                    milestone_details = st.session_state.milestone_details
                except Exception as e:
                    st.error(f"Error fetching milestone details: {e}")
                    milestone_details = {}
        else:
            milestone_details = st.session_state.milestone_details

        if milestone_details:
            # Ensure each text_area gets a unique key
            updated_milestone_description = st.text_area(
                "Milestone Description", 
                value=milestone_details["Description"], 
                height=100,
                key=f"milestone_desc_{selected_milestone}"  # Unique key
            )
            # Print milestone_details to understand its structure
            print(milestone_details)

            # Attempt to access the deadline value with proper error handling
            try:
                # Check if 'properties' exists and then access 'deadline'
                if "properties" in milestone_details:
                    deadline_value = milestone_details["properties"].get("deadline", None)
                else:
                    print("No 'properties' key found in milestone_details.")
                    deadline_value = None
            except KeyError as e:
                print(f"KeyError: {e}")
                deadline_value = None

            # If deadline_value is found and it's a string, parse it
            if isinstance(deadline_value, str):
                updated_milestone_deadline = st.date_input(
                    "Milestone Deadline", 
                    value=datetime.strptime(deadline_value, "%Y-%m-%d").date()
                )
            else:
                updated_milestone_deadline = st.date_input("Milestone Deadline", value=None)

            updated_milestone_status = st.selectbox(
                "Milestone Status",
                ["Not Started", "In Progress", "Completed", "On Hold", "Cancelled"],
                index=["Not Started", "In Progress", "Completed", "On Hold", "Cancelled"].index(milestone_details["Status"]),
                key=f"milestone_status_{selected_milestone}"  # Unique key
)

            tasks = milestone_details.get("Tasks", [])
            if tasks:
                selected_task = st.selectbox("Select Associated Task", tasks)
            else:
                # Fetch available tasks dynamically
                query = """
                    MATCH (t:Task)
                    RETURN t.name AS TaskName
                """
                available_tasks = user_graph.run(query).data()
                available_task_names = [task['TaskName'] for task in available_tasks]

                if available_task_names:
                    selected_task = st.selectbox("Select Task to Associate", available_task_names, help="Select a task to associate with this milestone.")
                    
                    if st.button("Associate Task"):
                        if selected_task:
                            associate_task_with_milestone(user_id, selected_project, selected_milestone, selected_task)
                else:
                    st.info("No tasks available to associate.")
        else:
            st.error("Milestone details could not be loaded.")
    else:
        # Creating a new milestone
        updated_milestone_name = st.text_input("Milestone Name")
        updated_milestone_description = st.text_area("Milestone Description", height=100)
        updated_milestone_deadline = st.date_input("Milestone Deadline")
        updated_milestone_status = st.selectbox(
            "Milestone Status",
            ["Not Started", "In Progress", "Completed", "On Hold", "Cancelled"],
            index=0
        )

    # Save Milestone Changes
    if st.button("💾 Save Milestone Changes"):
        if selected_milestone != "New Milestone":
            update_milestone(user_id, selected_project, selected_milestone, {
                "milestone_name": updated_milestone_name,
                "milestone_description": updated_milestone_description,
                "milestone_deadline": updated_milestone_deadline,
                "milestone_status": updated_milestone_status
            })
            st.success(f"✅ Milestone '{updated_milestone_name}' updated successfully!")
        else:
            create_milestone(user_id, selected_project, {
                "milestone_name": updated_milestone_name,
                "milestone_description": updated_milestone_description,
                "milestone_deadline": updated_milestone_deadline,
                "milestone_status": updated_milestone_status
            })
            st.success(f"✅ New milestone '{updated_milestone_name}' created successfully!")
# Add Task Section
st.markdown('<div class="section-header">📝 Assign Tasks</div>', unsafe_allow_html=True)
with st.expander("Add New Task"):
    task_name = st.text_input("Task Name")
    task_description = st.text_area("Task Description", height=100)
    
    task_assignee = st.selectbox("Assign to", get_team_members(user_id, selected_project))
    task_start_date = st.date_input("Task Start Date")
    task_deadline = st.date_input("Task Deadline")
    task_status = st.selectbox("Task Status", ["Not Started", "In Progress", "Completed"])

import streamlit as st
from neo4j import GraphDatabase
import asyncio
import concurrent.futures

import streamlit as st
from neo4j import GraphDatabase
import concurrent.futures

# Assuming user_graph is your Neo4j GraphDatabase connection
def check_user_and_project(user_id, project_name):
    """Check if the user and project exist."""
    query = """
        MATCH (u:User {id: $user_id}), (p:Project {name: $project_name})
        RETURN u, p
    """
    result = user_graph.run(query, user_id=user_id, project_name=project_name).data()
    if result:
        return True  # User and project exist
    return False  # Either user or project doesn't exist

def create_task_if_not_exists(task_name, task_description, start_date, due_date, assigned_to, status="Not Started"):
    """Check if task exists, if not create it."""
    query = """
        MERGE (t:Task {name: $task_name})
        ON CREATE SET t.description = $task_description, 
                      t.start_date = $start_date, 
                      t.due_date = $due_date, 
                      t.status = $status, 
                      t.assigned_to = $assigned_to
        RETURN t
    """
    result = user_graph.run(query, 
                            task_name=task_name, 
                            task_description=task_description, 
                            start_date=str(start_date), 
                            due_date=str(due_date), 
                            assigned_to=assigned_to, 
                            status=status).data()
    return result[0]["t"]  # Return created/merged task

def link_task_to_project(project_name, task_name):
    """Link the task to the project."""
    query = """
        MATCH (p:Project {name: $project_name}), (t:Task {name: $task_name})
        MERGE (p)-[:INCLUDES]->(t)
    """
    user_graph.run(query, project_name=project_name, task_name=task_name)

def add_task(user_id, project_name, task_name, task_description, start_date, due_date, assigned_to, status="Not Started"):
    """Add task to project step by step to avoid blocking."""
    try:
        # Step 1: Check if user and project exist
        if not check_user_and_project(user_id, project_name):
            st.error(f"❌ User or Project not found: '{user_id}' or '{project_name}'")
            return
        
        # Step 2: Create or merge the task
        task = create_task_if_not_exists(task_name, task_description, start_date, due_date, assigned_to, status)
        
        # Step 3: Link the task to the project
        link_task_to_project(project_name, task_name)
        
        st.success(f"✅ Task '{task_name}' assigned to {assigned_to}!")
    
    except Exception as e:
        st.error(f"Error occurred when adding task: {str(e)}")

# Streamlit button logic to call add_task function
if st.button("🗂️ Create Task"):
    try:
        task_assignee = task_assignee['MemberName'] if isinstance(task_assignee, dict) else task_assignee
        add_task(user_id, selected_project, task_name, task_description, task_start_date, task_deadline, task_assignee, task_status)
    except Exception as e:
        st.error(f"Failed to create task: {str(e)}")

import streamlit as st
import time

# Add Team Member Section
st.markdown('<div class="section-header">👥 Add Team Members</div>', unsafe_allow_html=True)
with st.expander("Add New Team Member"):
    new_member_name = st.text_input("Team Member Name")
    if st.button("➕ Add Member"):
        add_team_member(user_id, selected_project, new_member_name)
        st.success(f"✅ Member '{new_member_name}' added to the project!")

# Milestone and Task Progress Section
st.markdown('<div class="section-header">🗂️ Milestones & Tasks Progress</div>', unsafe_allow_html=True)
user_progress = get_user_progress(user_id, selected_project)
if user_progress:
    milestones = {}
    for entry in user_progress:
        milestone_name = entry.get('MilestoneName', 'Unknown Milestone')
        if milestone_name not in milestones:
            milestones[milestone_name] = []
        milestones[milestone_name].append({
            "TaskName": entry.get('TaskName', 'Unnamed Task'),
            "DueDate": entry.get('DueDate', 'No Due Date'),
            "Status": entry.get('TaskStatus', 'No Status')  # Default to 'No Status' if not found
        })

    for milestone, tasks in milestones.items():
        st.markdown(f"### Milestone: {milestone}")
        for task in tasks:
            col1, col2, col3 = st.columns([3, 1, 1])
            with col1:
                st.markdown(f"**Task:** {task['TaskName']}")
            with col2:
                st.markdown(f"**Due Date:** {task['DueDate']}")
            with col3:
                st.markdown(f"**Status:** {task['Status']}")
else:
    st.info("No milestones or tasks available yet.")

from datetime import datetime


# Add Update Task Section
st.markdown('<div class="section-header">📝 Update Task</div>', unsafe_allow_html=True)
with st.expander("Update Task Details"):
    # Fetch all task names for the selected project
    task_names = user_graph.run("""
        MATCH (u:User {id: $user_id})-[:TRACKS]->(p:Project {name: $project_name})-[:INCLUDES]->(t:Task)
        RETURN t.name AS TaskName
    """, user_id=user_id, project_name=selected_project).data()

    task_names = [task['TaskName'] for task in task_names]

    # Dropdown to select task name
    task_name_to_update = st.selectbox("Select Task to Update", task_names)

    if task_name_to_update:  # Only fetch details when a task is selected
        # Run the query to fetch task details
        task_details = user_graph.run("""
            MATCH (u:User {id: $user_id})-[:TRACKS]->(p:Project {name: $project_name})-[:INCLUDES]->(t:Task {name: $task_name})
            RETURN t.name AS TaskName, t.description AS Description, t.due_date AS DueDate, t.start_date AS StartDate, t.status AS Status, t.assigned_to AS Assignee
        """, user_id=user_id, project_name=selected_project, task_name=task_name_to_update).data()

        if task_details:  # If task is found
            task_detail = task_details[0]

            # Convert DueDate and StartDate strings to date objects if necessary
            if isinstance(task_detail["DueDate"], str):
                task_deadline = datetime.strptime(task_detail["DueDate"], "%Y-%m-%d").date()
            else:
                task_deadline = task_detail["DueDate"]

            if isinstance(task_detail["StartDate"], str):
                task_start_date = datetime.strptime(task_detail["StartDate"], "%Y-%m-%d").date()
            else:
                task_start_date = task_detail["StartDate"]

            # Display existing task details in the form fields
            task_name_updated = st.text_input("Task Name", task_detail["TaskName"])
            task_description_updated = st.text_area("Task Description", task_detail["Description"], height=100, key=f"description_{task_detail['TaskName']}")

            # Fetch the team members
            team_members = get_team_members(user_id, selected_project)

            # Extract the list of member names
            team_member_names = [member["MemberName"] for member in team_members]

            # Ensure assignee selection uses the right format
            task_assignee_updated = st.selectbox(
                "Assign to",
                team_member_names,  # Use the list of member names
                index=team_member_names.index(task_detail["Assignee"]) if task_detail["Assignee"] in team_member_names else 0,  # Find index based on Assignee
                key=f"assignee_{task_detail['TaskName']}"
            )
            # Use the corrected task_deadline and task_start_date values for the date_input
            task_deadline_updated = st.date_input("Task Deadline", value=task_deadline, key=f"deadline_{task_detail['TaskName']}")
            task_start_date_updated = st.date_input("Task Start Date", value=task_start_date, key=f"start_date_{task_detail['TaskName']}")
            # Ensure task_detail["Status"] is not None and provide a default value if it is
            task_status = task_detail["Status"] if task_detail["Status"] else "Not Started"

            task_status_updated = st.selectbox(
                "Task Status",
                ["Not Started", "In Progress", "Completed"],
                index=["Not Started", "In Progress", "Completed"].index(task_status),
                key=f"status_{task_detail['TaskName']}"
            )


            if st.button("💾 Save Task Changes"):
                updated_data = {
                    "name": task_name_updated,
                    "description": task_description_updated,
                    "assigned_to": task_assignee_updated,  # Ensure assignee is updated correctly
                    "due_date": str(task_deadline_updated),
                    "start_date": str(task_start_date_updated),  # Add start_date here
                    "status": task_status_updated
                }

                # Update the task in the database
                update_task(user_id, selected_project, task_name_updated, updated_data)
                st.success(f"✅ Task '{task_name_updated}' updated successfully!")
        
        else:
            st.info("Task not found. Please ensure the task name is correct.")

# Add Delete Task Section outside of the form
st.markdown('<div class="section-header">🗑️ Delete Task</div>', unsafe_allow_html=True)
with st.expander("Delete Task"):
    # Fetch all task names for the selected project
    task_names = user_graph.run("""
        MATCH (u:User {id: $user_id})-[:TRACKS]->(p:Project {name: $project_name})-[:INCLUDES]->(t:Task)
        RETURN t.name AS TaskName
    """, user_id=user_id, project_name=selected_project).data()

    task_names = [task['TaskName'] for task in task_names]

    # Dropdown to select task name
    task_name_to_delete = st.selectbox("Select Task to Delete", task_names)

    if task_name_to_delete:  # Only show delete option when a task is selected
        # Confirm task deletion
        if st.button(f"🗑️ Delete Task: {task_name_to_delete}"):
            with st.spinner(f"Deleting task '{task_name_to_delete}'..."):
                try:
                    delete_task(user_id, selected_project, task_name_to_delete)
                    st.warning(f"⚠️ Task '{task_name_to_delete}' has been deleted!")
                except Exception as e:
                    st.error(f"Error deleting task: {e}")

import streamlit as st

# Ensure that selected_project is initialized in session_state if not already
if 'selected_project' not in st.session_state:
    st.session_state.selected_project = None

# Display project selection dropdown
selected_project = st.selectbox(
    "Select Project", get_projects(user_id), index=get_projects(user_id).index(st.session_state.selected_project) if st.session_state.selected_project else 0
)

# When a project is selected, save it in session_state
if selected_project != st.session_state.selected_project:
    st.session_state.selected_project = selected_project
    st.success(f"Project '{selected_project}' selected. Now you can generate the report.")



# Function to delete project from the database
def delete_project(user_id, selected_project):
    query = """
        MATCH (u:User {id: $user_id})-[:TRACKS]->(p:Project {name: $selected_project})
        OPTIONAL MATCH (p)-[:HAS_MILESTONE]->(m:Milestone)
        OPTIONAL MATCH (m)-[:HAS_TASK]->(t:Task)
        DETACH DELETE p, m, t
        RETURN COUNT(p) AS project_deleted
    """
    try:
        result = user_graph.run(query, user_id=user_id, selected_project=selected_project).data()
        if result and result[0]["project_deleted"] > 0:
            st.success(f"✅ Project '{selected_project}' and associated milestones and tasks deleted successfully!")
        else:
            st.error(f"❌ Project '{selected_project}' not found or couldn't be deleted.")
    except Exception as e:
        st.error(f"❌ Error deleting project: {e}")

# Function to delete milestone from the database
def delete_milestone(user_id, selected_project, selected_milestone):
    query = """
        MATCH (u:User {id: $user_id})-[:TRACKS]->(p:Project {name: $selected_project})-[:HAS_MILESTONE]->(m:Milestone {name: $selected_milestone})
        OPTIONAL MATCH (m)-[:HAS_TASK]->(t:Task)
        DETACH DELETE m, t
        RETURN COUNT(m) AS milestone_deleted
    """
    try:
        result = user_graph.run(query, user_id=user_id, selected_project=selected_project, selected_milestone=selected_milestone).data()
        if result and result[0]["milestone_deleted"] > 0:
            st.success(f"✅ Milestone '{selected_milestone}' and associated tasks deleted successfully!")
        else:
            st.error(f"❌ Milestone '{selected_milestone}' not found or couldn't be deleted.")
    except Exception as e:
        st.error(f"❌ Error deleting milestone: {e}")

# Streamlit UI for Project and Milestone Deletion
st.markdown('<div class="section-header">🗑️ Delete Project / Milestone</div>', unsafe_allow_html=True)
def get_project_names(user_id):
    query = """
        MATCH (u:User {id: $user_id})-[:TRACKS]->(p:Project)
        RETURN p.name AS ProjectName
    """
    result = user_graph.run(query, user_id=user_id).data()
    
    # Extract the project names from the query result
    project_names = [project['ProjectName'] for project in result]
    
    return project_names

# Project Deletion
selected_project_for_deletion = st.selectbox("Select Project to Delete", get_project_names(user_id), help="Select a project to delete.")
if selected_project_for_deletion and st.button("🗑️ Delete Project"):
    with st.spinner(f"Deleting project '{selected_project_for_deletion}'..."):
        delete_project(user_id, selected_project_for_deletion)

# Milestone Deletion
selected_project_for_milestone = st.selectbox("Select Project to Delete Milestone From", get_project_names(user_id), help="Select a project to delete a milestone from.")
selected_milestone_for_deletion = st.selectbox("Select Milestone to Delete", get_milestone_names(user_id, selected_project_for_milestone), help="Select a milestone to delete.")
if selected_project_for_milestone and selected_milestone_for_deletion and st.button("🗑️ Delete Milestone"):
    with st.spinner(f"Deleting milestone '{selected_milestone_for_deletion}' from project '{selected_project_for_milestone}'..."):
        delete_milestone(user_id, selected_project_for_milestone, selected_milestone_for_deletion)

import streamlit as st
import plotly.graph_objects as go

# Assuming `get_user_progress`, `create_progress_chart`, `create_task_distribution`, `analyze_deviations`, etc. are defined somewhere in your code

# Display the selected project and provide the button to generate the report
if st.session_state.selected_project:
    # Display the selected project under the button
    st.markdown(f"### Selected Project: {st.session_state.selected_project}")

    # Display the 'Generate Progress Report' button only if a project is selected
    if st.button("Generate Progress Report"):
        selected_project = st.session_state.selected_project  # Use the selected project from session state

        if not selected_project:
            st.warning("Please select a project first.")
        else:
            # Execute the function and display the report output
            st.markdown("### Progress Report")
            report_output = answer_progress_and_recommendations(user_id, selected_project)
            if report_output:
                st.markdown(report_output)

                # Calculate KPIs for the selected project
                kpi_data = calculate_kpis(user_id, selected_project, get_user_progress(user_id, selected_project))  # Corrected this line

                # Ensure KPI data is valid
                if not kpi_data:
                    st.warning("KPI data is missing or incomplete.")
                else:
                    # Display KPI metrics
                    col1, col2, col3 = st.columns(3)
                    metrics = [
                        {"label": "Total Tasks", "value": kpi_data.get("Total Tasks", "N/A")},
                        {"label": "Completed", "value": kpi_data.get("Completed Tasks", "N/A")},
                        {"label": "Completion Rate", "value": f"{kpi_data.get('Completion Rate (%)', 'N/A')}%"}
                    ]

                    for col, metric in zip([col1, col2, col3], metrics):
                        with col:
                            st.markdown(f"""
                                <div class="metric-card">
                                    <div class="metric-value">{metric['value']}</div>
                                    <div class="metric-label">{metric['label']}</div>
                                </div>
                            """, unsafe_allow_html=True)

                    # # Display progress chart for the selected project
                    # if kpi_data:
                    #     fig = create_progress_chart(kpi_data, selected_project)  # Pass selected_project as the second argument
                    #     if fig:
                    #         fig.update_layout(
                    #             paper_bgcolor='rgba(0,0,0,0)',
                    #             plot_bgcolor='rgba(0,0,0,0)',
                    #             font={'color': '#2c3e50'}
                    #         )
                    #         st.plotly_chart(fig, use_container_width=True)
                    #     else:
                    #         st.warning("Progress chart could not be generated due to missing or incorrect data.")

                    # Display task distribution chart for the selected project
                    # st.markdown("### Task Distribution")
                    # task_progress = get_user_progress(user_id, selected_project)
                    # if not task_progress:
                    #     st.warning("No task progress data available.")
                    # else:
                    #     task_distribution_chart = create_task_distribution(task_progress, selected_project)  # Pass selected_project

                    #     if task_distribution_chart:
                    #         task_distribution_chart.update_layout(
                    #             paper_bgcolor='rgba(0,0,0,0)',
                    #             plot_bgcolor='rgba(0,0,0,0)',
                    #             font={'color': '#2c3e50'}
                    #         )
                    #         st.plotly_chart(task_distribution_chart, use_container_width=True)
                    #     else:
                    #         st.warning("Task distribution chart could not be generated.")

                    # Risk Analysis for the selected project
                    st.markdown("### ⚠️ Risk Analysis")
                    deviations = analyze_deviations(get_user_progress(user_id, selected_project))
                    if deviations:
                        for deviation in deviations:
                            st.markdown(f"""
                                <div class="status-badge status-warning">
                                    {deviation}
                                </div>
                            """, unsafe_allow_html=True)
                    else:
                        st.markdown(""" 
                            <div class="status-badge status-success">
                                ✨ All tasks are on track!
                            </div>
                        """, unsafe_allow_html=True)
    else:
        st.warning("Please confirm the selected project before generating the report.")

    # Footer
    st.markdown("""
        <div style="text-align: center; padding: 2rem 0; color: var(--text-secondary);">
            <p>Developed with ❤️ by Your Project Management Team</p>
            <p style="font-size: 0.8rem;">Version 2.0 | © 2024</p>
        </div>
    """, unsafe_allow_html=True)
