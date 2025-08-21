# tests/test_mother.py
import pytest
import json
import tempfile
import os
from unittest.mock import Mock, patch, MagicMock
import sys
sys.path.append('..')

from mother import MotherBrain, SelfImprovingAI, MetaLearner
from database import KnowledgeDB
from auth import UserManager, JWTManager
from cache import MotherCache

class TestMotherBrain:
    """Test suite for core MotherBrain functionality"""
    
    @pytest.fixture
    def mock_mother(self):
        """Create a MotherBrain instance with mocked dependencies"""
        with patch.dict(os.environ, {'GITHUB_FINE_GRAINED_PAT': 'github_pat_test123'}):
            with patch('mother.get_ai_heart') as mock_heart:
                mock_heart.return_value = None
                with patch('mother.Github') as mock_github:
                    mother = MotherBrain()
                    mother.knowledge = {
                        "_meta": {"version": "test", "timestamp": "2024-01-01"},
                        "0DAY:CVE-2023-1234": "Test exploit data",
                        "BUSINESS:AAPL": "Apple Inc stock data"
                    }
                    return mother
    
    def test_init_requires_github_token(self):
        """Test that initialization fails without GitHub token"""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(RuntimeError, match="GitHub token not configured"):
                MotherBrain()
    
    def test_validate_url_security(self, mock_mother):
        """Test URL validation prevents dangerous URLs"""
        # Valid URLs
        assert mock_mother._validate_url("https://example.com/api")
        assert mock_mother._validate_url("http://api.github.com")
        
        # Invalid URLs
        assert not mock_mother._validate_url("http://localhost/admin")
        assert not mock_mother._validate_url("https://127.0.0.1/secret")
        assert not mock_mother._validate_url("ftp://internal.company.com")
        assert not mock_mother._validate_url("invalid-url")
    
    def test_generate_exploit(self, mock_mother):
        """Test CVE exploit generation"""
        # Known CVE
        result = mock_mother.generate_exploit("CVE-2023-1234")
        assert "original" in result
        assert "mutated" in result
        assert "signature" in result
        assert result["original"] == "Test exploit data"
        
        # Unknown CVE
        result = mock_mother.generate_exploit("CVE-9999-0000")
        assert result == {"error": "Exploit not known"}
    
    def test_process_hacking_command_exploit(self, mock_mother):
        """Test hacking command processing - exploit"""
        result = mock_mother.process_hacking_command("exploit CVE-2023-1234")
        assert result["action"] == "exploit"
        assert result["target"] == "CVE-2023-1234"
        assert "exploit_data" in result
        
        # Test without target
        result = mock_mother.process_hacking_command("exploit")
        assert "error" in result
    
    def test_process_hacking_command_scan(self, mock_mother):
        """Test hacking command processing - scan"""
        result = mock_mother.process_hacking_command("scan network")
        assert result["action"] == "scan"
        assert result["type"] == "network"
        assert "commands" in result
        
        result = mock_mother.process_hacking_command("scan web")
        assert result["type"] == "web"
    
    def test_learn_url_with_mocked_requests(self, mock_mother):
        """Test URL learning with mocked HTTP responses"""
        with patch.object(mock_mother.session, 'get') as mock_get:
            # Mock successful JSON response
            mock_response = Mock()
            mock_response.json.return_value = {"CVE-2023-5678": "New vulnerability"}
            mock_response.raise_for_status.return_value = None
            mock_get.return_value = mock_response
            
            initial_count = len(mock_mother.knowledge)
            mock_mother._learn_url("https://example.com/cves.json", "cyber:0day")
            
            # Should process the JSON data
            mock_get.assert_called_once()
            assert mock_response.json.called
    
    def test_knowledge_processing(self, mock_mother):
        """Test knowledge processing from text"""
        # Test CVE processing
        cve_text = "CVE-2023-9999 is a critical vulnerability affecting Linux"
        mock_mother._process("cyber:0day", cve_text)
        assert "0DAY:CVE-2023-9999" in mock_mother.knowledge
        
        # Test business processing
        business_text = "TSLA stock rose 15% in Q4 2023"
        mock_mother._process("business", business_text)
        # Should find the Q4 pattern
        business_keys = [k for k in mock_mother.knowledge.keys() if k.startswith("BUSINESS:")]
        assert len(business_keys) > 1  # Original AAPL + new TSLA data

class TestSelfImprovingAI:
    """Test suite for self-improving AI functionality"""
    
    @pytest.fixture
    def temp_python_file(self):
        """Create temporary Python file for testing"""
        content = '''
import os
def vulnerable_function():
    user_input = input("Enter command: ")
    os.system(user_input)  # RCE vulnerability
    
def safe_function():
    return "Hello World"
'''
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(content)
            temp_path = f.name
        
        yield temp_path
        os.unlink(temp_path)
    
    def test_analyze_code_detects_vulnerabilities(self, temp_python_file):
        """Test that code analysis detects security vulnerabilities"""
        analyzer = SelfImprovingAI(temp_python_file)
        results = analyzer.analyze_code()
        
        assert 'vulnerabilities' in results
        assert 'suggestions' in results
        assert 'stats' in results
        
        # Should detect RCE vulnerability
        rce_vulns = [v for v in results['vulnerabilities'] if v['type'] == 'RCE']
        assert len(rce_vulns) > 0
        
        # Check severity calculation
        assert rce_vulns[0]['severity'] == 'critical'
    
    def test_vulnerability_solutions(self):
        """Test that solutions are provided for vulnerabilities"""
        analyzer = SelfImprovingAI()
        
        # Test all vulnerability types have solutions
        for vuln_type in analyzer.known_vulnerabilities.keys():
            solution = analyzer._get_solution(vuln_type)
            assert solution is not None
            assert len(solution) > 10  # Meaningful solution text

class TestDatabase:
    """Test suite for database functionality"""
    
    @pytest.fixture
    def temp_db(self):
        """Create temporary database for testing"""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            db_path = f.name
        
        db = KnowledgeDB(db_path)
        yield db
        os.unlink(db_path)
    
    def test_knowledge_storage_and_retrieval(self, temp_db):
        """Test storing and retrieving knowledge"""
        # Store knowledge
        temp_db.store_knowledge("test_key", "test_value", "test_domain")
        
        # Retrieve knowledge
        result = temp_db.get_knowledge("test_key")
        assert result == "test_value"
        
        # Test non-existent key
        result = temp_db.get_knowledge("nonexistent")
        assert result is None
    
    def test_knowledge_search(self, temp_db):
        """Test knowledge search functionality"""
        # Store test data
        temp_db.store_knowledge("CVE-2023-1234", "SQL injection vulnerability", "cyber")
        temp_db.store_knowledge("CVE-2023-5678", "XSS vulnerability", "cyber")
        temp_db.store_knowledge("STOCK:AAPL", "Apple stock data", "business")
        
        # Search by content
        results = temp_db.search_knowledge("vulnerability")
        assert len(results) == 2
        
        # Search by domain
        results = temp_db.search_knowledge("CVE", "cyber")
        assert len(results) == 2
        
        results = temp_db.search_knowledge("STOCK", "business")
        assert len(results) == 1
    
    def test_cve_storage(self, temp_db):
        """Test CVE-specific storage"""
        temp_db.store_cve("CVE-2023-1234", "exploit payload", 9.8)
        
        # Verify storage via direct query
        with temp_db.get_connection() as conn:
            result = conn.execute(
                "SELECT * FROM cve_exploits WHERE cve_id = ?", 
                ("CVE-2023-1234",)
            ).fetchone()
            
            assert result is not None
            assert result['payload'] == "exploit payload"
            assert result['severity'] == 9.8
    
    def test_query_logging(self, temp_db):
        """Test user query logging"""
        temp_db.log_query("test query", "test response", "192.168.1.1")
        
        with temp_db.get_connection() as conn:
            result = conn.execute("SELECT COUNT(*) as count FROM user_queries").fetchone()
            assert result['count'] == 1

class TestAuthentication:
    """Test suite for authentication system"""
    
    @pytest.fixture
    def temp_auth_db(self):
        """Create temporary database with auth tables"""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            db_path = f.name
        
        db = KnowledgeDB(db_path)
        user_manager = UserManager(db)
        yield user_manager
        os.unlink(db_path)
    
    def test_user_creation(self, temp_auth_db):
        """Test user creation"""
        result = temp_auth_db.create_user("testuser", "test@example.com", "password123")
        assert result["success"] is True
        assert "user_id" in result
        
        # Test duplicate username
        result = temp_auth_db.create_user("testuser", "test2@example.com", "password123")
        assert result["success"] is False
    
    def test_user_authentication(self, temp_auth_db):
        """Test user authentication"""
        # Create user
        temp_auth_db.create_user("testuser", "test@example.com", "password123")
        
        # Valid login
        user = temp_auth_db.authenticate_user("testuser", "password123", "192.168.1.1")
        assert user is not None
        assert user["username"] == "testuser"
        
        # Invalid password
        user = temp_auth_db.authenticate_user("testuser", "wrongpassword", "192.168.1.1")
        assert user is None
        
        # Invalid username
        user = temp_auth_db.authenticate_user("nonexistent", "password123", "192.168.1.1")
        assert user is None
    
    def test_api_key_generation(self, temp_auth_db):
        """Test API key generation and validation"""
        # Create user
        result = temp_auth_db.create_user("testuser", "test@example.com", "password123")
        user_id = result["user_id"]
        
        # Generate API key
        api_key = temp_auth_db.generate_api_key(
            user_id, "Test Key", ["read", "write"], 30
        )
        assert api_key.startswith("apg_")
        
        # Validate API key
        key_data = temp_auth_db.validate_api_key(api_key)
        assert key_data is not None
        assert key_data["username"] == "testuser"
        assert "read" in key_data["permissions"]
        assert "write" in key_data["permissions"]
        
        # Invalid API key
        key_data = temp_auth_db.validate_api_key("invalid_key")
        assert key_data is None
    
    def test_jwt_token_generation(self):
        """Test JWT token generation and verification"""
        jwt_manager = JWTManager("test_secret_key")
        
        user_data = {
            "id": 1,
            "username": "testuser",
            "role": "user"
        }
        
        # Generate token
        token = jwt_manager.generate_token(user_data)
        assert isinstance(token, str)
        assert len(token) > 50  # JWT tokens are typically long
        
        # Verify token
        decoded = jwt_manager.verify_token(token)
        assert decoded is not None
        assert decoded["username"] == "testuser"
        assert decoded["role"] == "user"
        
        # Invalid token
        decoded = jwt_manager.verify_token("invalid.token.here")
        assert decoded is None

class TestCache:
    """Test suite for caching functionality"""
    
    def test_cache_operations(self):
        """Test basic cache operations"""
        # Mock Redis unavailable scenario
        cache = MotherCache("redis://invalid:9999")
        assert cache.enabled is False
        
        # Test disabled cache
        assert cache.get("test_key") is None
        assert cache.set("test_key", "test_value") is False
        assert cache.delete("test_key") is False
    
    @patch('redis.from_url')
    def test_cache_with_mock_redis(self, mock_redis):
        """Test cache with mocked Redis"""
        # Setup mock Redis
        mock_client = Mock()
        mock_client.ping.return_value = True
        mock_client.get.return_value = b'{"data": "test_value"}'
        mock_client.setex.return_value = True
        mock_redis.return_value = mock_client
        
        cache = MotherCache()
        assert cache.enabled is True
        
        # Test operations
        result = cache.get("test_key")
        assert result == {"data": "test_value"}
        
        success = cache.set("test_key", {"data": "new_value"})
        assert success is True

class TestIntegration:
    """Integration tests for complete system"""
    
    @pytest.fixture
    def test_app(self):
        """Create test Flask app"""
        from mother import app
        app.config['TESTING'] = True
        app.config['JWT_MANAGER'] = JWTManager("test_secret")
        
        with app.test_client() as client:
            yield client
    
    def test_health_endpoint(self, test_app):
        """Test health check endpoint"""
        response = test_app.get('/health')
        assert response.status_code == 200
        
        data = json.loads(response.data)
        assert 'status' in data
        assert 'knowledge_items' in data
    
    def test_ask_endpoint(self, test_app):
        """Test knowledge query endpoint"""
        response = test_app.get('/ask?q=test_query')
        assert response.status_code == 200
        
        data = json.loads(response.data)
        assert 'query' in data
        assert 'result' in data
        
        # Test missing query parameter
        response = test_app.get('/ask')
        assert response.status_code == 400
    
    def test_exploit_endpoint(self, test_app):
        """Test exploit generation endpoint"""
        response = test_app.get('/exploit/CVE-2023-1234')
        assert response.status_code == 200
        
        # Test invalid CVE format
        response = test_app.get('/exploit/INVALID-FORMAT')
        assert response.status_code == 400

# Performance tests
class TestPerformance:
    """Performance and load testing"""
    
    def test_knowledge_search_performance(self):
        """Test search performance with large dataset"""
        import time
        
        # Create large test dataset
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            db_path = f.name
        
        try:
            db = KnowledgeDB(db_path)
            
            # Insert 1000 test records
            start_time = time.time()
            for i in range(1000):
                db.store_knowledge(f"TEST:{i}", f"Test data {i}", "test")
            insert_time = time.time() - start_time
            
            # Search performance
            start_time = time.time()
            results = db.search_knowledge("Test")
            search_time = time.time() - start_time
            
            # Performance assertions
            assert insert_time < 5.0  # Should insert 1000 records in < 5 seconds
            assert search_time < 1.0  # Should search in < 1 second
            assert len(results) == 20  # Limited to 20 results
