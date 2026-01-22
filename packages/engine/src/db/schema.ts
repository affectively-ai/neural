
import { dash } from "@buley/dash";

export async function initializeSchema() {
    console.log("Initializing Neural Schema...");

    // Neurons Table
    // id: UUID
    // type: input, hidden, output
    // bias: float
    // activation: string (tanh, relu, sigmoid)
    await dash.execute(`
        CREATE TABLE IF NOT EXISTS neurons (
            id TEXT PRIMARY KEY,
            type TEXT NOT NULL,
            bias REAL DEFAULT 0.0,
            activation TEXT DEFAULT 'tanh',
            created_at INTEGER DEFAULT (unixepoch())
        )
    `);

    // Synapses Table
    // id: UUID
    // from_id: neuron UUID
    // to_id: neuron UUID
    // weight: float
    await dash.execute(`
        CREATE TABLE IF NOT EXISTS synapses (
            id TEXT PRIMARY KEY,
            from_id TEXT NOT NULL,
            to_id TEXT NOT NULL,
            weight REAL DEFAULT 0.0,
            created_at INTEGER DEFAULT (unixepoch()),
            FOREIGN KEY(from_id) REFERENCES neurons(id),
            FOREIGN KEY(to_id) REFERENCES neurons(id)
        )
    `);

    console.log("Schema initialized.");
}
