import bcrypt

# Simple hardcoded user store: username -> hashed password
# In a real system, this would be a database table, not a dictionary in code.
USERS = {
    "sichu": bcrypt.hashpw(b"sichu123", bcrypt.gensalt()),
}

def verify_user(username, password):
    if username not in USERS:
        return False
    stored_hash = USERS[username]
    return bcrypt.checkpw(password.encode(), stored_hash)