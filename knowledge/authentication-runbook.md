# Authentication Service Runbook

The authentication service validates access tokens and communicates with the corporate identity provider.

## Deployment guardrails

Roll back an authentication deployment if the login error rate remains above 5 percent for 10 consecutive minutes or if p95 login latency remains above 800 milliseconds for 15 minutes.

## Token validation

Token validation allows 60 seconds of clock skew between the issuer and service hosts. During identity-provider signing-key rotation, the previous signing key must remain available for 24 hours so tokens issued shortly before rotation continue to validate.

## Escalation

Authentication incidents are owned by the Identity Platform on-call engineer. Security Operations must also be engaged when failures indicate invalid signatures, unexpected issuers, or suspected credential compromise.
