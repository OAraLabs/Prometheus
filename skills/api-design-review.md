---
name: api-design-review
description: "Use when reviewing API designs, OpenAPI specs, or endpoint patterns for REST conventions, breaking changes, consistency, security, and developer experience. Covers naming, versioning, pagination, error handling, auth patterns, and backward compatibility."
version: 1.0.0
author: alirezarezvani/claude-skills
license: MIT
---
<!-- Provenance: alirezarezvani/claude-skills | engineering/api-design-reviewer/SKILL.md | MIT -->

# API Design Review

Comprehensive API design analysis covering REST conventions, breaking change detection, consistency scoring, security patterns, and developer experience.

## API Design Scoring Dimensions

| Dimension | Weight | What to Evaluate |
|-----------|--------|-----------------|
| Consistency | 30% | Naming conventions, response patterns, structural consistency |
| Documentation | 20% | Completeness and clarity of API docs |
| Security | 20% | Authentication, authorization, security headers |
| Usability | 15% | Ease of use, discoverability, developer experience |
| Performance | 15% | Caching, pagination, efficiency patterns |

## Resource Naming Conventions

```
Good:
  /api/v1/users
  /api/v1/user-profiles
  /api/v1/orders/123/line-items

Bad:
  /api/v1/getUsers          (verb in URL)
  /api/v1/user_profiles     (underscores)
  /api/v1/orders/123/lineItems  (camelCase in path)
```

**Rules:** Kebab-case for URL path segments, camelCase for JSON fields, plural nouns for collections.

## HTTP Method Usage

| Method | Purpose | Idempotent | Safe |
|--------|---------|------------|------|
| GET | Retrieve resources | Yes | Yes |
| POST | Create new resources | No | No |
| PUT | Replace entire resource | Yes | No |
| PATCH | Partial update | Not necessarily | No |
| DELETE | Remove resource | Yes | No |

## URL Structure

```
Collection:    /api/v1/users
Individual:    /api/v1/users/123
Nested:        /api/v1/users/123/orders
Actions:       /api/v1/users/123/activate  (POST)
Filtering:     /api/v1/users?status=active&role=admin
Field select:  /api/v1/users?fields=id,name,email
```

## Versioning Strategies

| Strategy | Example | Pros | Cons |
|----------|---------|------|------|
| URL (recommended) | `/api/v1/users` | Clear, explicit, easy to route | URL proliferation |
| Header | `Accept: application/vnd.api+json;version=1` | Clean URLs | Less visible, harder to test |
| Media type | `Accept: application/vnd.myapi.v1+json` | RESTful | Complex to implement |
| Query param | `/api/users?version=1` | Simple | Not RESTful |

## Pagination Patterns

### Cursor-Based (preferred for large datasets)
```json
{
  "data": [...],
  "pagination": {
    "nextCursor": "eyJpZCI6MTIzfQ==",
    "hasMore": true
  }
}
```

### Offset-Based
```json
{
  "data": [...],
  "pagination": {
    "offset": 20, "limit": 10,
    "total": 150, "hasMore": true
  }
}
```

### Page-Based
```json
{
  "data": [...],
  "pagination": {
    "page": 3, "pageSize": 10,
    "totalPages": 15, "totalItems": 150
  }
}
```

## Error Response Format

```json
{
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "The request contains invalid parameters",
    "details": [
      {
        "field": "email",
        "code": "INVALID_FORMAT",
        "message": "Email address is not valid"
      }
    ],
    "requestId": "req-123456",
    "timestamp": "2024-02-16T13:00:00Z"
  }
}
```

### HTTP Status Code Usage

| Code | Meaning |
|------|---------|
| 400 | Invalid request syntax or parameters |
| 401 | Authentication required |
| 403 | Authenticated but not authorized |
| 404 | Resource not found |
| 409 | Resource conflict (duplicate, version mismatch) |
| 422 | Valid syntax but semantic errors |
| 429 | Rate limit exceeded |
| 500 | Unexpected server error |

## Authentication Patterns

| Pattern | Header | Use Case |
|---------|--------|----------|
| Bearer Token | `Authorization: Bearer <token>` | Web apps, SPAs |
| API Key | `X-API-Key: <key>` | Service integrations |
| OAuth 2.0 | `Authorization: Bearer <oauth-token>` | Third-party auth |
| mTLS | Certificate-based | Service-to-service |

## Rate Limiting

### Response Headers
```
X-RateLimit-Limit: 1000
X-RateLimit-Remaining: 999
X-RateLimit-Reset: 1640995200
```

### 429 Response
```json
{
  "error": {
    "code": "RATE_LIMIT_EXCEEDED",
    "message": "Too many requests",
    "retryAfter": 3600
  }
}
```

## Idempotency

Idempotent methods: GET, PUT, DELETE. POST is not idempotent.

For non-idempotent operations (e.g., payments), use idempotency keys:
```
POST /api/v1/payments
Idempotency-Key: 123e4567-e89b-12d3-a456-426614174000
```

## Breaking vs Non-Breaking Changes

### Safe Changes (Non-Breaking)
- Adding optional fields to requests
- Adding fields to responses
- Adding new endpoints
- Making required fields optional
- Adding new enum values (with graceful handling)

### Breaking Changes (Require Version Bump)
- Removing fields from responses
- Making optional fields required
- Changing field types
- Removing endpoints
- Changing URL structures
- Modifying error response formats

## HATEOAS

```json
{
  "id": "123",
  "name": "John Doe",
  "_links": {
    "self": { "href": "/api/v1/users/123" },
    "orders": { "href": "/api/v1/users/123/orders" },
    "deactivate": { "href": "/api/v1/users/123/deactivate", "method": "POST" }
  }
}
```

## Performance Patterns

### Caching
```
Cache-Control: public, max-age=3600
ETag: "123456789"
Last-Modified: Wed, 21 Oct 2015 07:28:00 GMT
```

### Efficiency
- Field selection (`?fields=id,name,email`)
- Compression (gzip)
- Efficient pagination
- ETags for conditional requests
- Batch operations to avoid N+1 queries
- Async processing for heavy operations

## Security Checklist

- Validate all input parameters; sanitize user data
- Use parameterized queries; implement request size limits
- HTTPS everywhere
- Secure token storage with expiration and refresh
- Principle of least privilege; resource-based permissions
- Audit access patterns

## Anti-Patterns to Avoid

1. **Verb-based URLs** -- use nouns for resources
2. **Inconsistent response formats** -- standardize structures
3. **Over-nesting** -- avoid deeply nested hierarchies (max 2-3 levels)
4. **Ignoring HTTP status codes** -- use appropriate codes
5. **Poor error messages** -- provide actionable, specific information
6. **Missing pagination** -- always paginate list endpoints
7. **No versioning strategy** -- plan for API evolution from day one
8. **Exposing internal structure** -- design for external consumption
9. **Missing rate limiting** -- protect from abuse and overload
10. **Inadequate testing** -- test error cases and edge conditions
