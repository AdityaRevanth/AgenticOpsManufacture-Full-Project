import asyncio
import json

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


def print_result(result):
    for content in result.content:
        if hasattr(content, "text"):
            try:
                data = json.loads(content.text)
                print(json.dumps(data, indent=2))
            except json.JSONDecodeError:
                print(content.text)
        else:
            print(content)


async def analyze_equipment(session):
    print("\n=== Equipment Analysis ===\n")

    equipment_id = input("Equipment ID: ")
    equipment_type = input("Equipment type: ")

    temperature_c = float(input("Temperature (°C): "))
    vibration_mm_s = float(input("Vibration (mm/s): "))
    pressure_psi = float(input("Pressure (psi): "))
    rpm = float(input("RPM: "))
    current_amps = float(input("Current (A): "))

    result = await session.call_tool(
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

    print("\n=== DIAGNOSTIC RESULT ===\n")
    print_result(result)


async def analyze_from_csv(session):
    print("\n=== CSV Equipment Analysis ===\n")

    csv_path = input(
        "CSV path (press Enter for sample_sensor_data.csv): "
    ).strip()

    if not csv_path:
        csv_path = "data/sample_sensor_data.csv"

    equipment_id = input("Equipment ID: ").strip()
    equipment_type = input("Equipment type: ").strip()

    result = await session.call_tool(
        "analyze_equipment_from_csv",
        {
            "equipment_id": equipment_id,
            "equipment_type": equipment_type,
            "csv_path": csv_path,
        },
    )

    print("\n=== CSV DIAGNOSTIC RESULT ===\n")
    print_result(result)


async def get_anomaly_score(session):
    print("\n=== Quick Anomaly Check ===\n")

    temperature_c = float(input("Temperature (°C): "))
    vibration_mm_s = float(input("Vibration (mm/s): "))
    pressure_psi = float(input("Pressure (psi): "))
    rpm = float(input("RPM: "))
    current_amps = float(input("Current (A): "))

    result = await session.call_tool(
        "get_anomaly_score",
        {
            "temperature_c": temperature_c,
            "vibration_mm_s": vibration_mm_s,
            "pressure_psi": pressure_psi,
            "rpm": rpm,
            "current_amps": current_amps,
        },
    )

    print("\n=== ANOMALY RESULT ===\n")
    print_result(result)


async def search_failure_history(session):
    print("\n=== Failure History Search ===\n")

    query = input("Describe the symptoms: ")

    equipment_type = input(
        "Equipment type (press Enter to skip): "
    ).strip()

    arguments = {
        "query": query,
    }

    if equipment_type:
        arguments["equipment_type"] = equipment_type

    result = await session.call_tool(
        "query_failure_history",
        arguments,
    )

    print("\n=== HISTORICAL MATCHES ===\n")
    print_result(result)


async def list_mcp_tools(session):
    print("\n=== Available MCP Tools ===\n")

    tools = await session.list_tools()

    for number, tool in enumerate(tools.tools, start=1):
        print(f"{number}. {tool.name}")
        print(f"   {tool.description}")
        print()


async def main():

    print("\n==========================================")
    print("   PLANT PREDICTIVE MAINTENANCE")
    print("          MCP CLIENT")
    print("==========================================\n")

    server_params = StdioServerParameters(
        command="python",
        args=["mcp_server/server.py"],
    )

    async with stdio_client(server_params) as (read, write):

        async with ClientSession(read, write) as session:

            await session.initialize()

            print("Connected to MCP server!\n")

            while True:

                print("==========================================")
                print("              MCP MENU")
                print("==========================================")
                print("1. Analyze equipment")
                print("2. Get anomaly score")
                print("3. Search failure history")
                print("4. Analyze equipment from CSV")
                print("5. List MCP tools")
                print("6. Exit")
                print("==========================================")

                choice = input("Select an option: ").strip()

                try:

                    if choice == "1":
                        await analyze_equipment(session)

                    elif choice == "2":
                        await get_anomaly_score(session)

                    elif choice == "3":
                        await search_failure_history(session)

                    elif choice == "4":
                        await analyze_from_csv(session)

                    elif choice == "5":
                        await list_mcp_tools(session)

                    elif choice == "6":
                        print("\nExiting MCP client...")
                        break

                    else:
                        print("\nInvalid option. Please choose 1-6.")

                except ValueError:
                    print(
                        "\nInvalid input. Please enter valid numbers."
                    )

                except Exception as e:
                    print(f"\nMCP request failed: {e}")

                print()


if __name__ == "__main__":
    asyncio.run(main())