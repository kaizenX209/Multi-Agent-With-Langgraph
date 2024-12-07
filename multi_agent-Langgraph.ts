import "dotenv/config";
import { END, Annotation, StateGraph, START } from "@langchain/langgraph";
import { HumanMessage, SystemMessage } from "@langchain/core/messages";
import { tool } from "@langchain/core/tools";
import { ChatOpenAI } from "@langchain/openai";
import { TavilySearchResults } from "@langchain/community/tools/tavily_search";
import { RunnableConfig } from "@langchain/core/runnables";
import { createReactAgent } from "@langchain/langgraph/prebuilt";
import { z } from "zod";
import {
  ChatPromptTemplate,
  MessagesPlaceholder,
} from "@langchain/core/prompts";

/**
 * Hệ thống đa tác tử (Multi-Agent) cho phép nhiều agent AI tương tác với nhau
 * để giải quyết các yêu cầu phức tạp của người dùng
 */
export class MultiAgentSystem {
  // Định nghĩa các agent trong hệ thống
  private members = ["researcher", "weather_expert"] as const;
  private llm: ChatOpenAI;
  private tavilyTool: TavilySearchResults;

  constructor() {
    // Khởi tạo model GPT-4 với temperature = 0 để có kết quả nhất quán
    this.llm = new ChatOpenAI({
      modelName: "gpt-4",
      temperature: 0,
    });
    // Công cụ tìm kiếm web Tavily
    this.tavilyTool = new TavilySearchResults();
  }

  /**
   * Thiết lập trạng thái ban đầu cho đồ thị agent
   * - messages: Lưu trữ lịch sử tin nhắn giữa các agent
   * - next: Xác định agent tiếp theo sẽ xử lý hoặc kết thúc (END)
   */
  private setupAgentState() {
    return Annotation.Root({
      messages: Annotation({
        reducer: (x: any, y: any) => x.concat(y),
        default: () => [],
      }),
      next: Annotation({
        reducer: (x: any, y: any) => y ?? x ?? END,
        default: () => END,
      }),
    });
  }

  /**
   * Tạo node Researcher có khả năng tìm kiếm thông tin trên web
   * sử dụng Tavily search engine
   */
  private createResearcherNode() {
    const researcherAgent = createReactAgent({
      llm: this.llm,
      tools: [this.tavilyTool],
      messageModifier: new SystemMessage(
        "You are a web researcher. You may use the Tavily search engine to search the web for" +
          " important information, so the Retriever in your team can retrieve the information."
      ),
    });

    return async (state: any, config?: RunnableConfig) => {
      const result = await researcherAgent.invoke(state, config);
      const lastMessage = result.messages[result.messages.length - 1];
      return {
        messages: [
          new HumanMessage({
            content: lastMessage.content,
            name: "Researcher",
          }),
        ],
      };
    };
  }

  /**
   * Tạo node Weather Expert có khả năng kiểm tra thời tiết cho một thành phố
   * thông qua tool get_weather
   */
  private createWeatherNode() {
    const weatherAgent = createReactAgent({
      llm: this.llm,
      tools: [
        tool(
          async (input) => {
            const city = input.city;
            console.log("----");
            console.log(`Checking weather for: ${city}`);
            console.log("----");
            return "Sunny!";
          },
          {
            name: "get_weather",
            description: "Get the current weather for a city.",
            schema: z.object({
              city: z.string().describe("The city to get weather for"),
            }),
          }
        ),
      ],
      messageModifier: new SystemMessage(
        "You are a weather expert. You help check weather conditions for different locations."
      ),
    });

    return async (state: any, config?: RunnableConfig) => {
      const result = await weatherAgent.invoke(state, config);
      const lastMessage = result.messages[result.messages.length - 1];
      return {
        messages: [
          new HumanMessage({
            content: lastMessage.content,
            name: "WeatherExpert",
          }),
        ],
      };
    };
  }

  /**
   * Tạo Supervisor chain điều phối luồng xử lý giữa các agent
   * - Quyết định agent nào sẽ xử lý tiếp theo
   * - Kết thúc quy trình khi hoàn thành (FINISH)
   */
  private async createSupervisorChain() {
    const systemPrompt =
      "You are a supervisor tasked with managing a conversation between the" +
      " following workers: {members}. Given the following user request," +
      " respond with the worker to act next. Each worker will perform a" +
      " task and respond with their results and status. When finished," +
      " respond with FINISH.";

    const options = [END, ...this.members] as const;

    const routingFunction = {
      name: "route",
      description: "Select the next role.",
      type: "function",
      parameters: {
        type: "object",
        properties: {
          next: {
            type: "string",
            enum: options,
            description: "The next role to run, or FINISH if we are done",
          },
        },
        required: ["next"],
      },
    };

    const prompt = ChatPromptTemplate.fromMessages([
      ["system", systemPrompt],
      new MessagesPlaceholder("messages"),
      [
        "system",
        "Given the conversation above, who should act next?" +
          " Or should we FINISH? Select one of: {options}",
      ],
    ]);

    const formattedPrompt = await prompt.partial({
      options: options.join(", "),
      members: this.members.join(", "),
    });

    const chain = formattedPrompt
      .pipe(
        this.llm.bind({
          functions: [routingFunction],
          function_call: { name: "route" },
        })
      )
      .pipe((output) => {
        const functionCall = output.additional_kwargs.function_call;
        if (!functionCall) {
          throw new Error("No function call returned");
        }
        try {
          return JSON.parse(functionCall.arguments);
        } catch (error) {
          console.error("Error parsing function arguments:", error);
          throw error;
        }
      });

    return chain;
  }

  /**
   * Thiết lập đồ thị luồng xử lý và xử lý yêu cầu
   * Luồng hoạt động:
   * 1. Bắt đầu từ Supervisor
   * 2. Supervisor chọn agent phù hợp (researcher/weather_expert)
   * 3. Agent được chọn xử lý và trả kết quả
   * 4. Quay lại Supervisor để quyết định bước tiếp theo
   * 5. Kết thúc khi Supervisor trả về FINISH
   */
  private async setupGraphAndProcess(query: string) {
    const workflow = new StateGraph(this.setupAgentState())
      .addNode("researcher", this.createResearcherNode())
      .addNode("weather_expert", this.createWeatherNode())
      .addNode("supervisor", await this.createSupervisorChain());

    this.members.forEach((member) => {
      workflow.addEdge(member, "supervisor");
    });

    workflow.addConditionalEdges("supervisor", (x: any) => x.next);
    workflow.addEdge(START, "supervisor");

    const graph = workflow.compile();

    const inputs = {
      messages: [
        new HumanMessage({
          content: query,
        }),
      ],
    };

    try {
      const messages: any[] = [];
      const events = await graph.streamEvents(inputs, {
        version: "v1",
      });

      for await (const event of events) {
        switch (event.event) {
          case "on_chat_model_stream":
            if (event.data?.chunk?.content) {
              console.log(`Token: ${event.data.chunk.content}`);
            }
            break;
          case "on_chain_stream":
            console.log(`\n=== ${event.name} ===`);
            console.log(event.data);
            break;
          case "on_chain_end":
            if (event.data?.output?.messages) {
              messages.push(...event.data.output.messages);
              console.log(`\nOutput from ${event.name}:`);
              console.log(event.data.output);
            }
            break;
        }
      }
      return messages;
    } catch (error) {
      console.error("Error during streaming:", error);
      throw error;
    }
  }

  /**
   * Phương thức public để xử lý câu hỏi từ người dùng
   * @param query Câu hỏi của người dùng
   */
  public async processQuery(query: string): Promise<any> {
    return await this.setupGraphAndProcess(query);
  }
}

/**
 * Hàm main để chạy thử nghiệm hệ thống
 * Ví dụ: Kiểm tra thời tiết ở San Francisco
 */
async function main() {
  const multiAgentSystem = new MultiAgentSystem();
  const messages = await multiAgentSystem.processQuery(
    "What is the weather in Hanoi?"
  );
  console.log("----");
  console.log("Final Output:", messages[messages.length - 1]?.content);
}

main().catch(console.error);
