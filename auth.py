# auth.py - Add this new file
import jwt
import bcrypt
from datetime import datetime, timedelta
from functools import wraps
from flask import request, jsonify, current_app
from typing import Dict, Optional, List
import secrets
import hashlib

class UserManager:
    def __init__(self, db_connection):
        self.db = db_connection
        self._init_auth_tables()
    
    def _init_auth_tables(self):
        """Create authentication tables"""
        with self.db.get_connection() as conn:
            conn.executescript('''
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT UNIQUE NOT NULL,
                    email TEXT UNIQUE NOT NULL,
                    password_hash TEXT NOT NULL,
                    role TEXT NOT NULL DEFAULT 'user',
                    is_active BOOLEAN DEFAULT 1,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_login TIMESTAMP
                );
                
                CREATE TABLE IF NOT EXISTS api_keys (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER REFERENCES users(id),
                    key_hash TEXT UNIQUE NOT NULL,
                    name TEXT NOT NULL,
                    permissions TEXT,  -- JSON array of permissions
                    is_active BOOLEAN DEFAULT 1,
                    expires_at TIMESTAMP,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_used TIMESTAMP
                );
                
                CREATE TABLE IF NOT EXISTS login_attempts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT,
                    ip_address TEXT,
                    success BOOLEAN,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                
                -- Create default admin user
                INSERT OR IGNORE INTO users (username, email, password_hash, role)
                VALUES ('admin', 'admin@americanpowerai.com', ?, 'admin');
            ''', (self._hash_password('change_me_immediately'),))
    
    def _hash_password(self, password: str) -> str:
        """Hash password using bcrypt"""
        return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
    
    def _verify_password(self, password: str, hash: str) -> bool:
        """Verify password against hash"""
        return bcrypt.checkpw(password.encode('utf-8'), hash.encode('utf-8'))
    
    def create_user(self, username: str, email: str, password: str, role: str = 'user') -> Dict:
        """Create new user"""
        try:
            password_hash = self._hash_password(password)
            with self.db.get_connection() as conn:
                conn.execute('''
                    INSERT INTO users (username, email, password_hash, role)
                    VALUES (?, ?, ?, ?)
                ''', (username, email, password_hash, role))
                
                user_id = conn.lastrowid
                return {"success": True, "user_id": user_id}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def authenticate_user(self, username: str, password: str, ip_address: str) -> Optional[Dict]:
        """Authenticate user and return user data"""
        # Check rate limiting
        if self._is_rate_limited(username, ip_address):
            return None
        
        try:
            with self.db.get_connection() as conn:
                user = conn.execute('''
                    SELECT id, username, email, password_hash, role, is_active
                    FROM users WHERE username = ? AND is_active = 1
                ''', (username,)).fetchone()
                
                success = False
                if user and self._verify_password(password, user['password_hash']):
                    success = True
                    # Update last login
                    conn.execute('''
                        UPDATE users SET last_login = CURRENT_TIMESTAMP
                        WHERE id = ?
                    ''', (user['id'],))
                
                # Log attempt
                conn.execute('''
                    INSERT INTO login_attempts (username, ip_address, success)
                    VALUES (?, ?, ?)
                ''', (username, ip_address, success))
                
                if success:
                    return dict(user)
                
        except Exception as e:
            print(f"Authentication error: {e}")
        
        return None
    
    def _is_rate_limited(self, username: str, ip_address: str) -> bool:
        """Check if user/IP is rate limited"""
        with self.db.get_connection() as conn:
            # Check failed attempts in last 15 minutes
            failed_attempts = conn.execute('''
                SELECT COUNT(*) as count FROM login_attempts
                WHERE (username = ? OR ip_address = ?)
                AND success = 0
                AND timestamp > datetime('now', '-15 minutes')
            ''', (username, ip_address)).fetchone()
            
            return failed_attempts['count'] >= 5
    
    def generate_api_key(self, user_id: int, name: str, permissions: List[str], expires_days: int = 30) -> str:
        """Generate API key for user"""
        # Generate secure key
        api_key = f"apg_{secrets.token_urlsafe(32)}"
        key_hash = hashlib.sha256(api_key.encode()).hexdigest()
        
        expires_at = datetime.utcnow() + timedelta(days=expires_days)
        
        with self.db.get_connection() as conn:
            conn.execute('''
                INSERT INTO api_keys (user_id, key_hash, name, permissions, expires_at)
                VALUES (?, ?, ?, ?, ?)
            ''', (user_id, key_hash, name, json.dumps(permissions), expires_at))
        
        return api_key
    
    def validate_api_key(self, api_key: str) -> Optional[Dict]:
        """Validate API key and return permissions"""
        if not api_key.startswith('apg_'):
            return None
        
        key_hash = hashlib.sha256(api_key.encode()).hexdigest()
        
        with self.db.get_connection() as conn:
            result = conn.execute('''
                SELECT ak.permissions, ak.user_id, u.username, u.role
                FROM api_keys ak
                JOIN users u ON ak.user_id = u.id
                WHERE ak.key_hash = ? 
                AND ak.is_active = 1 
                AND (ak.expires_at IS NULL OR ak.expires_at > CURRENT_TIMESTAMP)
            ''', (key_hash,)).fetchone()
            
            if result:
                # Update last used
                conn.execute('''
                    UPDATE api_keys SET last_used = CURRENT_TIMESTAMP
                    WHERE key_hash = ?
                ''', (key_hash,))
                
                return {
                    "user_id": result['user_id'],
                    "username": result['username'],
                    "role": result['role'],
                    "permissions": json.loads(result['permissions'])
                }
        
        return None

class JWTManager:
    def __init__(self, secret_key: str):
        self.secret_key = secret_key
        self.algorithm = 'HS256'
        self.token_expiry = timedelta(hours=24)
    
    def generate_token(self, user_data: Dict) -> str:
        """Generate JWT token"""
        payload = {
            'user_id': user_data['id'],
            'username': user_data['username'],
            'role': user_data['role'],
            'exp': datetime.utcnow() + self.token_expiry,
            'iat': datetime.utcnow()
        }
        
        return jwt.encode(payload, self.secret_key, algorithm=self.algorithm)
    
    def verify_token(self, token: str) -> Optional[Dict]:
        """Verify and decode JWT token"""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            return payload
        except jwt.ExpiredSignatureError:
            return None
        except jwt.InvalidTokenError:
            return None

# Authentication decorators
def token_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = request.headers.get('Authorization')
        if not token:
            return jsonify({'error': 'Token missing'}), 401
        
        if token.startswith('Bearer '):
            token = token[7:]
        
        try:
            jwt_manager = current_app.config['JWT_MANAGER']
            data = jwt_manager.verify_token(token)
            if not data:
                return jsonify({'error': 'Invalid token'}), 401
            
            request.current_user = data
        except Exception as e:
            return jsonify({'error': 'Token validation failed'}), 401
        
        return f(*args, **kwargs)
    return decorated

def api_key_required(permissions: List[str] = None):
    def decorator(f):
        @wraps(f)
        def decorated(*args, **kwargs):
            api_key = request.headers.get('X-API-Key')
            if not api_key:
                return jsonify({'error': 'API key missing'}), 401
            
            try:
                user_manager = current_app.config['USER_MANAGER']
                key_data = user_manager.validate_api_key(api_key)
                if not key_data:
                    return jsonify({'error': 'Invalid API key'}), 401
                
                # Check permissions
                if permissions:
                    user_perms = key_data.get('permissions', [])
                    if not any(perm in user_perms for perm in permissions):
                        return jsonify({'error': 'Insufficient permissions'}), 403
                
                request.current_user = key_data
            except Exception as e:
                return jsonify({'error': 'API key validation failed'}), 401
            
            return f(*args, **kwargs)
        return decorated
    return decorator

def role_required(required_role: str):
    def decorator(f):
        @wraps(f)
        def decorated(*args, **kwargs):
            if not hasattr(request, 'current_user'):
                return jsonify({'error': 'Authentication required'}), 401
            
            user_role = request.current_user.get('role', 'user')
            role_hierarchy = {'user': 0, 'moderator': 1, 'admin': 2}
            
            if role_hierarchy.get(user_role, 0) < role_hierarchy.get(required_role, 0):
                return jsonify({'error': 'Insufficient privileges'}), 403
            
            return f(*args, **kwargs)
        return decorated
    return decorator

# Update mother.py with authentication
@app.route('/auth/login', methods=['POST'])
def login():
    data = request.get_json()
    username = data.get('username')
    password = data.get('password')
    
    if not username or not password:
        return jsonify({'error': 'Username and password required'}), 400
    
    user_manager = current_app.config['USER_MANAGER']
    user = user_manager.authenticate_user(username, password, request.remote_addr)
    
    if user:
        jwt_manager = current_app.config['JWT_MANAGER']
        token = jwt_manager.generate_token(user)
        return jsonify({
            'token': token,
            'user': {
                'username': user['username'],
                'role': user['role']
            }
        })
    else:
        return jsonify({'error': 'Invalid credentials'}), 401

@app.route('/auth/api-key', methods=['POST'])
@token_required
@role_required('admin')
def create_api_key():
    data = request.get_json()
    name = data.get('name', 'API Key')
    permissions = data.get('permissions', ['read'])
    expires_days = data.get('expires_days', 30)
    
    user_manager = current_app.config['USER_MANAGER']
    api_key = user_manager.generate_api_key(
        request.current_user['user_id'],
        name,
        permissions,
        expires_days
    )
    
    return jsonify({'api_key': api_key})

# Protected route examples
@app.route('/admin/users', methods=['GET'])
@token_required
@role_required('admin')
def list_users():
    return jsonify({'users': 'admin only data'})

@app.route('/learn', methods=['POST'])
@api_key_required(['write', 'admin'])
def learn():
    # Existing learn functionality with API key protection
    mother.learn_all()
    return jsonify({
        "status": "Knowledge updated",
        "user": request.current_user['username']
    })

# Add to requirements.txt:
# PyJWT>=2.8.0
# bcrypt>=4.0.0
