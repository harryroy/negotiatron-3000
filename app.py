import os
import json
import math
import random
from flask import Flask, render_template, request, jsonify, Response
from openai import OpenAI
import time
from datetime import datetime

# ------------------------------------------------------------------------------
# CONFIGURATION AND PLACEHOLDERS
# ------------------------------------------------------------------------------

# Replace with your actual OpenAI API key or other provider's key as needed.
OPENAI_API_KEY = "YOUR-KEY-HERE"

# For demonstration, we pretend to have "gpt-4o-mini" endpoints. In reality,
# you can use any other LLM.
USER_PERSONA_MODEL = "gpt-4o-mini"
OPPONENT_PERSONA_MODEL = "gpt-4o-mini"
SCORING_MODEL = "gpt-4o-mini"

# We limit the depth of the conversation tree to 5 moves ahead.
MAX_DEPTH = 5

# Limit total expansions per node (to avoid massive tree blow-up).
# In a real app, you might want to dynamically vary or sample more expansions.
BRANCHING_FACTOR = 3

# Maximum total tree expansions per step (to control costs/performance).
MAX_API_CALLS_PER_STEP = 750

# Add these variables with other constants
CURRENT_SIMULATION = None
CURRENT_DEPTH = 3  # Default depth
NODES_PROCESSED = 0  # Initialize at module level
TOTAL_EXPECTED_NODES = 0  # Initialize at module level

# Calculate expected nodes dynamically
def calculate_expected_nodes(depth=CURRENT_DEPTH, branching=BRANCHING_FACTOR):
    """Calculate expected number of nodes in the tree"""
    return sum(branching ** i for i in range(depth + 1))

def reset_progress_tracking():
    """Reset progress tracking variables"""
    global NODES_PROCESSED, TOTAL_EXPECTED_NODES
    NODES_PROCESSED = 0
    TOTAL_EXPECTED_NODES = calculate_expected_nodes()

def update_progress(increment=1):
    """Update progress tracking in a consistent way"""
    global NODES_PROCESSED
    NODES_PROCESSED += increment
    return NODES_PROCESSED

app = Flask(__name__)

# ------------------------------------------------------------------------------
# HELPER DATA STRUCTURES
# ------------------------------------------------------------------------------

class ConversationNode:
    """
    A node in the conversation tree, representing a state of the conversation.
    
    Attributes:
        conversation (list): A list of (speaker, text) tuples.
        probability (float): Probability of arriving at this node from its parent.
        children (list): Child nodes in the conversation tree.
        score (float): The final score for this node (only relevant at leaf or
                       computed after expansions).
    """
    _next_id = 0  # Class-level variable to generate unique IDs

    def __init__(self, conversation, probability=1.0):
        self.id = ConversationNode._next_id  # Assign unique ID
        ConversationNode._next_id += 1
        self.conversation = conversation  # list of (speaker, text)
        self.probability = probability
        self.children = []
        self.score = None

    def add_child(self, child_node):
        self.children.append(child_node)

    @classmethod
    def reset_id_counter(cls):
        cls._next_id = 0

# ------------------------------------------------------------------------------
# LLM CALLS (SIMPLIFIED PLACEHOLDERS)
# ------------------------------------------------------------------------------

# Initialize OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY)

def log_event(message, event_type="info"):
    """Generate a log event"""
    timestamp = datetime.now().strftime("%H:%M:%S")
    return f"data: {json.dumps({
        'type': event_type,
        'timestamp': timestamp,
        'message': message
    })}\n\n"

def call_persona_model(persona_model, conversation_history, speaker, goal, context):
    """
    Call OpenAI API to generate next possible messages.
    Returns a list of (message_text, probability_estimate) tuples.
    """
    # Format conversation history for the prompt
    formatted_history = "\n".join([
        f"{turn[0]}: {turn[1]}" for turn in conversation_history
    ])
    
    # Create system prompt based on speaker
    if speaker == "User":
        system_prompt = (
            "You are helping generate multiple possible user responses or actions. "
            f"The user's goal is: {goal}\n"
            f"Context: {context}\n"
            "Generate exactly 3 different possible responses or actions that could help achieve this goal. Make sure each response or action takes a different approach from each other, so that then entire range of options is explored. Include some options that might be seem unwise, radical or self destructive."
            "Format each response or action on a new line starting with 'RESPONSE:'. "
            "Each response or action should be different in approach or tone."
        )
    else:
        system_prompt = (
            "You are simulating possible opponent responses in a conversation. "
            f"Context: {context}\n"
            "Generate exactly 3 different plausible responses or actionsthe opponent might give. Make sure each response or action takes a different approach from each other, so that then entire range of options is explored. Include some options that might be seem unwise, radical or self destructive."
            "Format each response or action on a new line starting with 'RESPONSE:'. "
            "Vary the responses or actions in terms of receptiveness and tone."
        )

    try:
        # Log before API call
        log_message = f"Generating responses for {speaker}..."
        
        # Make API call with high temperature for variety
        response = client.chat.completions.create(
            model="gpt-4o-mini",  # or "gpt-3.5-turbo" for lower cost
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Current conversation:\n{formatted_history}\n\nGenerate 3 possible next messages or actions for {speaker}."}
            ],
            temperature=0.9,
            n=1
        )
        
        # Process API response
        content = response.choices[0].message.content.strip()
        
        messages = [
            line.strip().replace("RESPONSE:", "").strip()
            for line in content.split("\n")
            if line.strip().startswith("RESPONSE:")
        ]
        
        # If no properly formatted responses found, try to extract any non-empty lines
        if not messages:
            messages = [line.strip() for line in content.split("\n") if line.strip()]
            # Take up to 3 lines
            messages = messages[:3]
        
        # If still no messages, provide a fallback
        if not messages:
            messages = [
                f"I understand what you're saying.",
                f"Could you tell me more about that?",
                f"Let me think about this for a moment."
            ]
        
        # Assign equal probabilities
        num_responses = len(messages)
        probability = 1.0 / num_responses
        
        # Return the responses with probabilities
        return log_message, [(msg, probability) for msg in messages]

    except Exception as e:
        error_message = f"Error generating responses: {str(e)}"
        # Return fallback responses
        return error_message, [
            ("I understand what you're saying.", 1.0/3.0),
            ("Could you tell me more about that?", 1.0/3.0),
            ("Let me think about this for a moment.", 1.0/3.0)
        ]

def call_scoring_model(conversation, goal, context, scoring_prompt=None):
    """
    Call OpenAI API to score a conversation path.
    Returns a float score between 1 and 10.
    """
    # Format conversation for scoring
    formatted_conversation = "\n".join([
        f"{turn[0]}: {turn[1]}" for turn in conversation
    ])
    
    # Use custom prompt if provided, otherwise use default
    if not scoring_prompt:
        scoring_prompt = (
            "You are an expert negotiation coach evaluating a negotiation. "
            "Score the conversation or actions on how well it achieves the user's goal. "
            "Consider factors like maintaining relationship, clear communication, and progress toward the goal. "
            "Return only a number between 1 and 10, where 10 is perfect."
        )
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": scoring_prompt},
                {"role": "user", "content": f"Goal: {goal}\nContext: {context}\n\nConversation:\n{formatted_conversation}\n\nScore (1-10):"}
            ],
            temperature=0.3,
            max_tokens=10
        )
        
        # Extract number from response
        score_text = response.choices[0].message.content.strip()
        try:
            score = float(score_text)
            # Ensure score is between 1 and 10
            score = max(1.0, min(10.0, score))
            return score
        except ValueError:
            return 5.0  # Default score if we can't parse the response
            
    except Exception as e:
        print(f"Error in scoring: {str(e)}")
        return 5.0  # Default score on error

# ------------------------------------------------------------------------------
# MONTE CARLO TREE EXPANSION
# ------------------------------------------------------------------------------

def generate_progress_events():
    """Generator function for SSE events"""
    global NODES_PROCESSED, TOTAL_EXPECTED_NODES
    reset_progress_tracking()
    last_percentage = -1
    
    yield log_event("Starting conversation tree analysis...", "start")
    
    while NODES_PROCESSED < TOTAL_EXPECTED_NODES:
        current_percentage = int((NODES_PROCESSED / TOTAL_EXPECTED_NODES) * 100)
        if current_percentage > last_percentage:
            yield f"data: {json.dumps({'type': 'progress', 'value': current_percentage})}\n\n"
            last_percentage = current_percentage
            # Add a log message every 20%
            if current_percentage % 20 == 0:
                yield log_event(f"Processing: {current_percentage}% complete", "info")
        time.sleep(0.1)
    
    yield log_event("Analysis complete!", "complete")

@app.route("/progress")
def progress():
    return Response(generate_progress_events(), mimetype="text/event-stream")

def build_conversation_tree(current_node, depth, goal):
    """
    Recursively build a conversation tree up to CURRENT_DEPTH.
    """
    yield log_event(f"=== ENTERING build_conversation_tree at depth {depth} ===", "debug")
    yield log_event(f"Current conversation length: {len(current_node.conversation)}", "debug")
    
    if depth >= CURRENT_DEPTH:
        update_progress()
        yield log_event(f"Reached depth {CURRENT_DEPTH}, stopping branch", "info")
        return
    
    # If we're at depth 0, just emit a single node for the entire prior conversation.
    if depth == 0 and current_node.conversation:
        last_speaker, last_message = current_node.conversation[-1]
        yield f"data: {json.dumps({
            'type': 'tree_node',
            'message': last_message,
            'speaker': last_speaker,
            'depth': 0,
            'branch_index': 0,
            'parent_index': None
        })}\n\n"
    
    current_len = len(current_node.conversation)
    if current_len % 2 == 0:
        next_speaker = "User"
        model = USER_PERSONA_MODEL
    else:
        next_speaker = "Opponent"
        model = OPPONENT_PERSONA_MODEL
    
    yield log_event(f"\nExpanding node at depth {depth} for {next_speaker}", "info")
    
    try:
        yield log_event(f"Calling persona model for {next_speaker} at depth {depth}", "debug")
        
        # Call the model and get both log message and expansions
        log_message, expansions = call_persona_model(
            persona_model=model,
            conversation_history=current_node.conversation,
            speaker=next_speaker,
            goal=goal,
            context=CURRENT_SIMULATION['context']
        )
        
        # Log the message from call_persona_model
        yield log_event(log_message, "info")
        
        # Log each response
        for i, (msg, prob) in enumerate(expansions, 1):
            yield log_event(f"Response option {i}: {msg}", "info")
        
        yield log_event(f"Generated {len(expansions)} responses for {next_speaker}", "info")
        
        # Limit to BRANCHING_FACTOR
        expansions = expansions[:BRANCHING_FACTOR]
        
        for i, (msg_text, msg_prob) in enumerate(expansions):
            yield log_event(f"Creating branch {i+1}/{len(expansions)} at depth {depth}", "debug")
            
            child_conversation = current_node.conversation + [(next_speaker, msg_text)]
            child_prob = current_node.probability * msg_prob
            
            child_node = ConversationNode(
                conversation=child_conversation,
                probability=child_prob
            )
            
            current_node.add_child(child_node)
            update_progress()  # Use the centralized function
            
            # Send tree node data
            node_data = {
                'type': 'tree_node', 
                'message': msg_text, 
                'speaker': next_speaker, 
                'depth': depth + 1,
                'branch_index': i, 
                'parent_index': current_node.id
            }
            yield log_event(f"Sending node data for depth {depth+1}, branch {i}", "debug")
            yield f"data: {json.dumps(node_data)}\n\n"
            
            # Recursively build the tree for this child
            yield log_event(f"Starting recursion for branch {i+1}/{len(expansions)} at depth {depth+1}", "debug")
            for recursive_item in build_conversation_tree(child_node, depth + 1, goal):
                yield recursive_item
            yield log_event(f"Completed recursion for branch {i+1}/{len(expansions)} at depth {depth+1}", "debug")
        
        yield log_event(f"Completed all {len(expansions)} branches at depth {depth}", "debug")
            
    except Exception as e:
        yield log_event(f"Error at depth {depth}: {str(e)}", "error")
        import traceback
        yield log_event(f"Traceback: {traceback.format_exc()}", "error")
    
    yield log_event(f"=== EXITING build_conversation_tree at depth {depth} ===", "debug")

def score_conversation_tree(node, goal):
    """
    Recursively score all nodes in the tree.
    Returns the score for the current node.
    """
    # If it's a leaf node or has no children, score the conversation directly
    if not node.children:
        node.score = call_scoring_model(node.conversation, goal, f"Current conversation: {node.conversation}")
        return node.score
    
    # Otherwise, score children first
    child_scores = []
    for child in node.children:
        child_score = score_conversation_tree(child, goal)
        # For opponent turns, use probability-weighted scores
        if len(node.conversation) % 2 == 1:  # Opponent's turn
            child_scores.append(child_score * child.probability)
        else:  # User's turn
            child_scores.append(child_score)  # Don't weight user's choices
    
    # Node's score depends on whose turn it is
    if len(node.conversation) % 2 == 0:  # User's turn
        # Take the maximum possible score for user's choices
        node.score = max(child_scores) if child_scores else 5.0
    else:  # Opponent's turn
        # Take the weighted average for opponent's likely responses
        node.score = sum(child_scores) / len(child_scores) if child_scores else 5.0
    
    return node.score

# ------------------------------------------------------------------------------
# TOP-LEVEL LOGIC TO FIND BEST USER MOVES
# ------------------------------------------------------------------------------

def find_best_user_moves(goal, conversation_history):
    """
    Build conversation tree and find the best moves for the user.
    """
    # Create root node with current conversation
    root = ConversationNode(conversation=conversation_history)
    
    # Reset node counter for progress tracking
    global NODES_PROCESSED
    NODES_PROCESSED = 0
    
    # Build the full conversation tree
    yield log_event("Starting to build conversation tree...", "info")
    
    # First, send the root node's conversation as initial nodes
    for i, (speaker, message) in enumerate(conversation_history):
        yield f"data: {json.dumps({
            'type': 'tree_node',
            'message': message,
            'speaker': speaker,
            'depth': i,
            'branch_index': 0,
            'parent_index': i-1 if i > 0 else None
        })}\n\n"
    
    # Build the tree and collect all possible future conversations
    tree_generator = build_conversation_tree(root, 0, goal)
    for item in tree_generator:
        yield item
    
    # After tree is built, find best moves
    yield log_event("Tree built, evaluating paths...", "info")
    
    # Score the entire tree from bottom up
    yield log_event("Starting to score all paths...", "info")
    score_conversation_tree(root, goal)
    
    # Find best immediate moves (direct children of root)
    if root.children:
        moves_with_scores = []
        for child in root.children:
            score = child.score if child.score is not None else 5.0  # Default to 5 if no score
            yield log_event(f"Evaluating move: {child.conversation[-1][1][:50]}... Score: {score:.2f}", "debug")
            moves_with_scores.append((child.conversation[-1][1], score))
        
        # Sort by score
        moves_with_scores.sort(key=lambda x: x[1], reverse=True)
        yield log_event(f"Found {len(moves_with_scores)} possible moves", "info")
        yield moves_with_scores[:3]  # Return top 3 moves
    else:
        yield log_event("No moves found!", "warning")
        yield []

def find_best_path(node):
    """
    Recursively find the single highest scoring path in the tree,
    returning a list of node IDs in that path from root to leaf.
    """
    if not node.children:
        # It's a leaf node
        return [node.id]

    # Otherwise, pick the child whose 'score' is highest.
    # (Adjust logic if you want to handle the "Opponent's turn" differently.)
    best_child = max(node.children, key=lambda c: c.score if c.score is not None else 0)
    return [node.id] + find_best_path(best_child)

# ------------------------------------------------------------------------------
# UTILITY
# ------------------------------------------------------------------------------

def random_probability_distribution(n):
    """
    Return a random list of n probabilities that sum to 1.
    """
    rand_nums = [random.random() for _ in range(n)]
    total = sum(rand_nums)
    return [x / total for x in rand_nums]

# ------------------------------------------------------------------------------
# FLASK ROUTES AND WEB UI
# ------------------------------------------------------------------------------

@app.route("/")
def index():
    return render_template("index.html")

@app.route('/simulate', methods=['POST'])
def simulate():
    """
    Endpoint to start the simulation process
    """
    global CURRENT_SIMULATION
    
    # Reset the node ID counter
    ConversationNode.reset_id_counter()
    
    # Reset progress tracking
    reset_progress_tracking()
    
    goal = request.form.get('goal')
    context = request.form.get('context')
    conversation_history = json.loads(request.form.get('conversation_history', '[]'))
    scoring_prompt = request.form.get('scoring_prompt')
    
    # Store the simulation parameters globally
    CURRENT_SIMULATION = {
        'goal': goal,
        'context': context,
        'conversation_history': conversation_history,
        'root': ConversationNode(conversation=conversation_history),
        'scoring_prompt': scoring_prompt
    }
    
    return jsonify({'status': 'success'})

@app.route('/stream')
def stream():
    """Stream simulation events back to the client"""
    def generate():
        if not CURRENT_SIMULATION:
            yield log_event('No simulation in progress', 'error')
            return
            
        yield log_event('Starting conversation tree analysis...', 'start')
        
        # Send initial progress
        yield f"data: {json.dumps({'type': 'progress', 'value': 0})}\n\n"
        
        # Build and evaluate tree
        root = CURRENT_SIMULATION['root']
        goal = CURRENT_SIMULATION['goal']
        
        yield log_event('Building conversation tree...', 'info')
        
        # First build the tree
        for item in build_conversation_tree(root, 0, goal):
            if isinstance(item, str):
                yield item
        
        yield log_event('Tree built, calculating scores...', 'info')
        
        # Then score the tree
        score_conversation_tree(root, goal)
        
        # Send highest-scoring path
        best_path = find_best_path(root)
        yield f"data: {json.dumps({'type': 'best_path', 'path': best_path})}\n\n"
        
        yield log_event('Finding best moves...', 'info')
        
        # Find best moves (direct children of root)
        moves_with_scores = []
        if root.children:
            for child in root.children:
                score = child.score if child.score is not None else 5.0
                moves_with_scores.append((child.conversation[-1][1], score))
                yield log_event(f'Found move with score {score:.2f}: {child.conversation[-1][1][:50]}...', 'debug')
            
            # Sort by score
            moves_with_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Send completion progress
        yield f"data: {json.dumps({'type': 'progress', 'value': 100})}\n\n"
        
        # Send results
        yield f"data: {json.dumps({
            'type': 'result',
            'data': [{'message': msg, 'expected_score': score} for msg, score in moves_with_scores[:3]]
        })}\n\n"
        
        yield log_event('Analysis complete!', 'complete')
        
    return Response(generate(), mimetype='text/event-stream')

@app.route('/update_depth', methods=['POST'])
def update_depth():
    global CURRENT_DEPTH, TOTAL_EXPECTED_NODES
    data = request.get_json()
    CURRENT_DEPTH = int(data.get('depth', 5))
    TOTAL_EXPECTED_NODES = calculate_expected_nodes(CURRENT_DEPTH)
    return jsonify({'status': 'success', 'depth': CURRENT_DEPTH, 'expected_nodes': TOTAL_EXPECTED_NODES})

# ------------------------------------------------------------------------------
# MAIN
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    # Run in debug mode for local development.
    app.run(debug=True)
