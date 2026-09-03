# Billing Service Runbook

The billing service receives payment-provider webhooks and turns them into idempotent internal billing events.

## Webhook delivery

Failed webhook deliveries are retried after approximately 1, 5, 15, and 60 minutes. Before manually replaying an event, verify its provider event ID has not already been processed. The event ID is the idempotency key.

## Degraded operation

If the payment provider is reachable but delivery is delayed, keep the service online and monitor the webhook backlog. Escalate when the oldest unprocessed webhook is more than 30 minutes old.

## Recovery

After provider recovery, allow automatic retries to drain the queue before manually replaying failures. Manual replay should be limited to events that remain failed after the normal retry schedule.
