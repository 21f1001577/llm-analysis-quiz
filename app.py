from flask import Flask, request, jsonify
import logging
import time
import traceback
from config import Config
from quiz_solver import QuizSolver
from logging.handlers import RotatingFileHandler
import os





# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config.from_object(Config)

@app.route('/', methods=['GET'])
def home():
    """Health check endpoint"""
    return jsonify({
        'status': 'running',
        'service': 'LLM Analysis Quiz Solver',
        'version': '1.0.0'
    }), 200

@app.route('/quiz', methods=['POST'])
def quiz_endpoint():
    """Main quiz endpoint that receives and processes quiz tasks"""
    start_time = time.time()
    
    try:
        # Validate JSON payload
        if not request.is_json:
            logger.warning('Invalid request: Not JSON')
            return jsonify({'error': 'Invalid JSON payload'}), 400
        
        data = request.get_json(silent=True)
        if data is None:
            return jsonify({"error": "Invalid JSON"}), 400
        
        # Validate required fields
        required_fields = ['email', 'secret', 'url']
        missing_fields = [field for field in required_fields if field not in data]
        
        if missing_fields:
            logger.warning(f'Missing fields: {missing_fields}')
            return jsonify({'error': f'Missing required fields: {", ".join(missing_fields)}'}), 400
        
        # Verify secret
        if data['secret'] != app.config['STUDENT_SECRET']:
            logger.warning(f'Invalid secret for email: {data["email"]}')
            return jsonify({'error': 'Invalid secret'}), 403
        
        # Verify email
        if data['email'] != app.config['STUDENT_EMAIL']:
            logger.warning(f'Invalid email: {data["email"]}')
            return jsonify({'error': 'Invalid email'}), 403
        
        logger.info(f'Received quiz request for URL: {data["url"]}')
        
        # Initialize quiz solver
        from config import Config
        solver = QuizSolver(Config())        
        # Process the quiz
        result = solver.solve_quiz_chain(data['url'], data['email'], data['secret'])
        
        elapsed_time = time.time() - start_time
        logger.info(f'Quiz processing completed in {elapsed_time:.2f} seconds')
        
        return jsonify({
            'status': 'success',
            'message': 'Quiz processing initiated',
            'elapsed_time': elapsed_time
        }), 200
        
    except Exception as e:
        logger.error(f'Error processing quiz: {str(e)}')
        logger.error(traceback.format_exc())
        return jsonify({
            'error': 'Internal server error',
            'message': str(e)
        }), 500

if __name__ == '__main__':
    import os
    
    # Create necessary folders
    os.makedirs(app.config['DOWNLOAD_FOLDER'], exist_ok=True)
    os.makedirs(app.config['TEMP_FOLDER'], exist_ok=True)
    
    # Run the application
    port = app.config['PORT']
    debug = app.config['DEBUG']
    
    logger.info(f'Starting LLM Analysis Quiz Solver on port {port}')
    app.run(host='0.0.0.0', port=port, debug=debug)