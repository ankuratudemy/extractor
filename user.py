from flask import Flask, request, jsonify
from azure.identity import DefaultAzureCredential
from azure.mgmt.apimanagement import ApiManagementClient
from azure.mgmt.apimanagement.models import UserContract

app = Flask(__name__)

# Azure API Management details
resource_group_name = "extractor"
service_name = "extractor-apim-service"

# Authenticate using Azure Managed Identity
credential = DefaultAzureCredential()
api_management_client = ApiManagementClient(credential, "https://management.azure.com")

# In-memory storage for users and API keys (for demonstration purposes)
users = {}
api_keys = {}

@app.route('/register', methods=['POST'])
def register_user():
    data = request.get_json()

    user_id = data.get('user_id')
    first_name = data.get('first_name')
    last_name = data.get('last_name')
    email = data.get('email')

    if not user_id or not first_name or not last_name or not email:
        return jsonify({'error': 'Missing required information'}), 400

    # Create a user in Azure API Management
    user = UserContract(user_id=user_id, first_name=first_name, last_name=last_name, email=email)

    api_management_client.user.create_or_update(resource_group_name, service_name, user_id, user)

    # Generate API key for the user
    api_key_name = f"{user_id}-api-key"
    api_key = api_management_client.user.generate_sso_url(resource_group_name, service_name, user_id, api_key_name)

    # Store user and API key information in memory (in a production scenario, use a database)
    users[user_id] = {'first_name': first_name, 'last_name': last_name, 'email': email}
    api_keys[user_id] = api_key

    return jsonify({'user_id': user_id, 'api_key': api_key}), 201

@app.route('/verify', methods=['POST'])
def verify_api_key():
    data = request.get_json()

    user_id = data.get('user_id')
    client_api_key = data.get('api_key')

    if not user_id or not client_api_key:
        return jsonify({'error': 'Missing required information'}), 400

    # Verify API key using Azure API Management
    try:
        api_management_client.access.get(resource_group_name, service_name, client_api_key)
        return jsonify({'verification_status': 'success', 'user_info': users.get(user_id, {})})
    except Exception as e:
        return jsonify({'verification_status': 'failed', 'error_message': str(e)}), 401

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)
