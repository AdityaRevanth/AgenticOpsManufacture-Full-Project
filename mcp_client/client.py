import asyncio
import json

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


async def call_mcp_tool(tool_name, arguments):
    server_params = StdioServerParameters(
        command="python",
        args=["mcp_server/server.py"],
    )

    async with stdio_client(server_params) as (read, write):

        async with ClientSession(read, write) as session:

            await session.initialize()

            result = await session.call_tool(
                tool_name,
                arguments,
            )

            for content in result.content:

                if hasattr(content, "text"):

                    try:
                        return json.loads(content.text)

                    except json.JSONDecodeError:
                        return content.text

            return None


def analyze_equipment(
    equipment_id,
    equipment_type,
    temperature_c,
    vibration_mm_s,
    pressure_psi,
    rpm,
    current_amps,
):
    return asyncio.run(
        call_mcp_tool(
            "analyze_equipment_readings",
            {
                "equipment_id": equipment_id,
                "equipment_type": equipment_type,
                "temperature_c": temperature_c,
                "vibration_mm_s": vibration_mm_s,
                "pressure_psi": pressure_psi,
                "rpm": rpm,
                "current_amps": current_amps,
            },
        )
    )