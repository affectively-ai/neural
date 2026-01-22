import { expect, test, describe, mock, beforeAll } from "bun:test";
import { NeuronRepository, SynapseRepository } from "./repository";

// Mock @buley/dash
const mockDash = {
    execute: mock((query, params) => {
        // Simple mock implementation
        if (query.includes("INSERT")) return Promise.resolve();
        if (query.includes("SELECT * FROM neurons")) return Promise.resolve([
            { id: "n1", type: "input", bias: 0.1, activation: "tanh" }
        ]);
        if (query.includes("SELECT * FROM synapses")) return Promise.resolve([
            { id: "s1", from_id: "n1", to_id: "n2", weight: 0.5 }
        ]);
        return Promise.resolve([]);
    }),
    addWithEmbedding: mock(() => Promise.resolve())
};

// Mock module
mock.module("@buley/dash", () => ({
    dash: mockDash
}));

describe("NeuronRepository", () => {
    test("create() executes INSERT query", async () => {
        const repo = new NeuronRepository();
        await repo.create({ id: "n1", type: "input", bias: 0.1, activation: "tanh" });
        expect(mockDash.execute).toHaveBeenCalled();
        const call = mockDash.execute.mock.calls[0];
        expect(call[0]).toContain("INSERT INTO neurons");
        expect(call[1]).toEqual(["n1", "input", 0.1, "tanh"]);
    });

    test("createWithSemantics() calls addWithEmbedding", async () => {
        const repo = new NeuronRepository();
        await repo.createWithSemantics(
            { id: "n2", type: "hidden", bias: 0, activation: "relu" },
            "detects curves"
        );
        expect(mockDash.addWithEmbedding).toHaveBeenCalledWith("n2", "detects curves");
    });

    test("getAll() returns neurons", async () => {
        mockDash.execute.mockClear();
        const repo = new NeuronRepository();
        const results = await repo.getAll();
        expect(results.length).toBe(1);
        expect(results[0].id).toBe("n1");
    });
});

describe("SynapseRepository", () => {
    test("create() executes INSERT query", async () => {
        const repo = new SynapseRepository();
        await repo.create({ id: "s1", from_id: "n1", to_id: "n2", weight: 0.5 });
        expect(mockDash.execute).toHaveBeenCalled();
        // Check latest call
    });

    test("getAll() returns synapses", async () => {
        const repo = new SynapseRepository();
        const results = await repo.getAll();
        expect(results.length).toBe(1);
        expect(results[0].weight).toBe(0.5);
    });
});
