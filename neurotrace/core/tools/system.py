import platform
from datetime import datetime

import requests
from dotenv import load_dotenv
from langchain.tools import BaseTool
from langchain_tavily import TavilySearch

from neurotrace.core.tools.factory import generic_tool_factory

# Load environment variables
load_dotenv()


def get_current_datetime(_: str = "") -> str:
    """
    Use this tool to get the current date and time in a human-readable format.
    :param _:
    :return:
    """
    return datetime.now().strftime("%A, %B %d, %Y at %I:%M %p")


def get_ip_address(_: str = "") -> str:
    """
    Use this tool to get the current public IP address.
    :param _:
    :return:
    """
    try:
        return requests.get("https://api.ipify.org").text
    except Exception:
        return "Unable to fetch IP address."


def get_location_by_ip(_: str = "") -> str:
    """
    Use this tool to get the current location based on the public IP address.
    :param _:
    :return:
    """
    try:
        ip = requests.get("https://api.ipify.org").text
        response = requests.get(f"https://ipinfo.io/{ip}/json").json()
        city = response.get("city", "")
        region = response.get("region", "")
        country = response.get("country", "")
        return f"{city}, {region}, {country}".strip(", ")
    except Exception:
        return "Unable to determine location."


def get_device_info(_: str = "") -> str:
    """
    Use this tool to get basic information about the current device.
    :param _:
    :return:
    """
    return (
        f"System: {platform.system()}\n"
        f"Release: {platform.release()}\n"
        f"Processor: {platform.processor()}\n"
        f"Machine: {platform.machine()}"
    )


def get_day_of_week(_: str = "") -> str:
    """
    Use this tool to get the current day of the week.
    :param _:
    :return:
    """
    return datetime.now().strftime("%A")


# ============= EXTERNAL TOOLS =============

web_search_tool = TavilySearch(
    max_results=5,
    topic="general",
    # include_answer=False,
    # include_raw_content=False,
    # include_images=False,
    # include_image_descriptions=False,
    # include_favicon=False,
    # search_depth="basic",
    # time_range="day",
    # include_domains=None,
    # exclude_domains=None,
    # country=None
)


def get_system_tools_list() -> list[BaseTool]:
    system_tools = [
        generic_tool_factory(
            tool_name="get_current_datetime",
            tool_description="Use this tool to get the current date and time in a human-readable format.",
            func=get_current_datetime,
        ),
        generic_tool_factory(
            tool_name="get_ip_address",
            tool_description="Use this tool to get the current public IP address.",
            func=get_ip_address,
        ),
        generic_tool_factory(
            tool_name="get_location_by_ip",
            tool_description="Use this tool to get the current location based on the public IP address.",
            func=get_location_by_ip,
        ),
        generic_tool_factory(
            tool_name="get_device_info",
            tool_description="Use this tool to get basic information about the current device.",
            func=get_device_info,
        ),
        generic_tool_factory(
            tool_name="get_day_of_week",
            tool_description="Use this tool to get the current day of the week.",
            func=get_day_of_week,
        ),
    ]

    # Add external tools like TavilySearch
    system_tools.extend(
        [
            generic_tool_factory(
                func=web_search_tool.run,
                tool_name="web_search",
                tool_description="Use this tool to perform a web search and retrieve relevant information.",
            )
        ]
    )
    return system_tools
