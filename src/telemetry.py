"""
telemetry.py — OpenTelemetry Configuration (Prompt P24)
======================================================

Sets up the OpenTelemetry TracerProvider with OTLP exporting
to Jaeger, incorporating resource attributes like service.name
and deployment.env.
"""

from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter

from src.settings import get_settings


def setup_telemetry(service_name: str, endpoint: str) -> TracerProvider:
    """Configure OpenTelemetry tracing globally."""
    settings = get_settings()

    resource = Resource.create({
        "service.name": service_name,
        "service.version": "1.0.0",
        "deployment.env": settings.ENV,
    })

    provider = TracerProvider(resource=resource)
    
    if endpoint:
        otlp_exporter = OTLPSpanExporter(endpoint=endpoint, insecure=True)
        span_processor = BatchSpanProcessor(otlp_exporter)
        provider.add_span_processor(span_processor)

    trace.set_tracer_provider(provider)
    return provider
