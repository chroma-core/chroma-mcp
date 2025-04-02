# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2024-04-02

### Added
- New `delete_document` tool for removing documents from collections
- New `chroma_update_documents` tool for updating existing documents
- Docker deployment support with Dockerfile
- Smithery configuration for deployment
- Environment variable support in Smithery config

### Changed
- Improved error handling across tools
- Removed sequential thinking in favor of more direct operations
- SSL parsing improvements and fixes

### Security
- Enhanced SSL handling and security configurations

## [0.1.11] - 2025-02-21

### Changed
- Version bump

## [0.1.10] - 2024-02-21

### Added
- Initial release
- Support for ephemeral, persistent, HTTP, and cloud Chroma clients
- Collection management tools
- Document operations (add, query, get)
- Claude Desktop integration
- Environment variable support
- Dotenv file support

### Security
- SSL support for HTTP and cloud clients
- Authentication support for HTTP clients
- API key management for cloud deployments 